from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:
    import config
    from core.ai.frame_sampler import EventFrameSampler, FrameCandidate, frame_diff_energy, iter_sampled_frames
    from core.ai.heuristics import FrameSignal, box_center_in_roi, classify_event
    from core.ai.qwen_client import QwenClientError, QwenVLClient
    from core.ai.yolo_detector import YoloDetector
    from core.db_ai import AIResultRecord, event_hash, event_key_from_event
    from core.models import Event
except ModuleNotFoundError:
    from teslacam_viewer_py import config
    from teslacam_viewer_py.core.ai.frame_sampler import (
        EventFrameSampler,
        FrameCandidate,
        frame_diff_energy,
        iter_sampled_frames,
    )
    from teslacam_viewer_py.core.ai.heuristics import FrameSignal, box_center_in_roi, classify_event
    from teslacam_viewer_py.core.ai.qwen_client import QwenClientError, QwenVLClient
    from teslacam_viewer_py.core.ai.yolo_detector import YoloDetector
    from teslacam_viewer_py.core.db_ai import AIResultRecord, event_hash, event_key_from_event
    from teslacam_viewer_py.core.models import Event


@dataclass(slots=True)
class AnalyzeOutput:
    record: AIResultRecord
    durations_ms: dict[str, int]


class EventAnalyzer:
    def __init__(self) -> None:
        self.detector = YoloDetector()
        self.qwen_client = QwenVLClient()
        self.frame_sampler = EventFrameSampler()
        self.model_version = self.detector.model_name if self.detector.available else "no_yolo"
        self.version = f"{config.AI_ANALYZER_VERSION}+{config.AI_RULE_VERSION}+{self.model_version}"

    def analyze_event(self, event: Event, roi_map: dict[str, dict[str, float]]) -> AnalyzeOutput:
        key = event_key_from_event(event)
        hash_value = event_hash(event, config.AI_RULE_VERSION, self.model_version)
        durations_ms: dict[str, int] = {}

        if cv2 is None:
            return AnalyzeOutput(
                record=AIResultRecord(
                    event_key=key,
                    version=self.version,
                    primary_type="other",
                    secondary_types=[],
                    risk_level=0,
                    objects={"person": 0, "vehicle": 0, "animal": 0},
                    ai_summary="缺少 OpenCV，无法执行本地视觉分析。",
                    evidence=[],
                    analyzer_source="rules",
                    status="error",
                    error_message="未安装 opencv-python",
                    event_hash=hash_value,
                ),
                durations_ms=durations_ms,
            )

        frame_signals, frame_candidates = self._collect_local_signals(event, roi_map, durations_ms)
        local = classify_event(event.timestamp, frame_signals, sampling_fps=config.AI_SAMPLING_FPS)

        record = AIResultRecord(
            event_key=key,
            version=self.version,
            primary_type=local.primary_type,
            secondary_types=local.secondary_types,
            risk_level=local.risk_level,
            objects=local.objects,
            ai_summary=local.ai_summary,
            evidence=local.evidence,
            analyzer_source="yolo+rules" if self.detector.available else "rules",
            status="ok",
            error_message="",
            event_hash=hash_value,
        )

        if not local.qwen_candidate:
            return AnalyzeOutput(record=record, durations_ms=durations_ms)

        if not self.qwen_client.available():
            record.status = "skipped"
            record.error_message = "未配置 DASHSCOPE_API_KEY，已跳过大模型分析"
            return AnalyzeOutput(record=record, durations_ms=durations_ms)

        if not frame_candidates:
            record.status = "skipped"
            record.error_message = "未提取到有效关键帧，已跳过大模型分析"
            return AnalyzeOutput(record=record, durations_ms=durations_ms)

        sampled = self.frame_sampler.sample_for_qwen(frame_candidates, max_images=config.AI_MAX_IMAGES)
        if not sampled:
            record.status = "skipped"
            record.error_message = "关键帧抽取失败，已跳过大模型分析"
            return AnalyzeOutput(record=record, durations_ms=durations_ms)

        image_paths = [img.path for img in sampled]
        hint = {
            "primary_type": local.primary_type,
            "risk_level": local.risk_level,
            "objects": local.objects,
            "metrics": local.metrics,
        }
        raw_error = ""
        try:
            for attempt in range(max(0, config.AI_JSON_RETRY) + 1):
                try:
                    parsed, _raw = self.qwen_client.analyze_images(
                        image_paths=image_paths,
                        local_hint=hint,
                        force_retry=attempt > 0,
                    )
                    record.primary_type = parsed.primary_type
                    record.secondary_types = parsed.secondary_types
                    record.risk_level = parsed.risk_level
                    record.ai_summary = parsed.ai_summary or record.ai_summary
                    if parsed.objects:
                        record.objects = parsed.objects
                    if parsed.evidence:
                        record.evidence = parsed.evidence
                    record.analyzer_source = f"{record.analyzer_source}+qwen"
                    record.status = "ok"
                    record.error_message = ""
                    return AnalyzeOutput(record=record, durations_ms=durations_ms)
                except QwenClientError as exc:
                    raw_error = str(exc)
            record.status = "error"
            record.error_message = f"大模型结果解析失败，已降级本地规则: {raw_error}"
            return AnalyzeOutput(record=record, durations_ms=durations_ms)
        finally:
            self.frame_sampler.cleanup(sampled)

    def _collect_local_signals(
        self,
        event: Event,
        roi_map: dict[str, dict[str, float]],
        durations_ms: dict[str, int],
    ) -> tuple[list[FrameSignal], list[FrameCandidate]]:
        signals: list[FrameSignal] = []
        candidates: list[FrameCandidate] = []
        for cam, clip in event.clips.items():
            roi = roi_map.get(cam, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
            duration = self._video_duration_ms(clip.path)
            durations_ms[str(clip.path.resolve())] = duration
            prev_gray = None
            for t_ms, frame in iter_sampled_frames(clip.path, config.AI_SAMPLING_FPS, config.AI_WINDOW_SECONDS):
                frame_h, frame_w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                diff = frame_diff_energy(prev_gray, gray)
                prev_gray = gray

                dets = self.detector.detect(frame)
                person_count = 0
                vehicle_count = 0
                animal_count = 0
                person_max_area = 0.0
                vehicle_max_area = 0.0
                person_in_roi = False
                vehicle_in_roi = False
                frame_area = max(1.0, float(frame_w * frame_h))

                for det in dets:
                    area_ratio = det.area / frame_area
                    if det.label == "person":
                        person_count += 1
                        person_max_area = max(person_max_area, area_ratio)
                        person_in_roi = person_in_roi or box_center_in_roi(
                            det.x1, det.y1, det.x2, det.y2, frame_w, frame_h, roi
                        )
                    elif det.label == "vehicle":
                        vehicle_count += 1
                        vehicle_max_area = max(vehicle_max_area, area_ratio)
                        vehicle_in_roi = vehicle_in_roi or box_center_in_roi(
                            det.x1, det.y1, det.x2, det.y2, frame_w, frame_h, roi
                        )
                    elif det.label == "animal":
                        animal_count += 1

                signal = FrameSignal(
                    cam=cam,
                    t_ms=t_ms,
                    brightness=brightness,
                    diff_energy=diff,
                    person_count=person_count,
                    vehicle_count=vehicle_count,
                    animal_count=animal_count,
                    person_max_area_ratio=person_max_area,
                    vehicle_max_area_ratio=vehicle_max_area,
                    person_in_roi=person_in_roi,
                    vehicle_in_roi=vehicle_in_roi,
                )
                signals.append(signal)
                self._append_frame_candidates(candidates, signal, frame)
        return signals, candidates

    def _append_frame_candidates(
        self,
        candidates: list[FrameCandidate],
        signal: FrameSignal,
        frame: Any,
    ) -> None:
        img = self._resize_for_qwen(frame)
        if signal.person_max_area_ratio > 0:
            candidates.append(
                FrameCandidate(
                    cam=signal.cam,
                    t_ms=signal.t_ms,
                    frame=img.copy(),
                    score=signal.person_max_area_ratio * 1000.0,
                    why="closest person",
                )
            )
        if signal.vehicle_max_area_ratio > 0:
            candidates.append(
                FrameCandidate(
                    cam=signal.cam,
                    t_ms=signal.t_ms,
                    frame=img.copy(),
                    score=signal.vehicle_max_area_ratio * 1000.0,
                    why="closest vehicle",
                )
            )
        if signal.diff_energy >= max(6.0, config.AI_IMPACT_DIFF_THRESHOLD * 0.5):
            candidates.append(
                FrameCandidate(
                    cam=signal.cam,
                    t_ms=signal.t_ms,
                    frame=img.copy(),
                    score=signal.diff_energy,
                    why="impact-like motion spike",
                )
            )

    @staticmethod
    def _video_duration_ms(path: Path) -> int:
        if cv2 is None:
            return 0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps and fps > 0 and frame_count and frame_count > 0:
                return int((frame_count / fps) * 1000)
            return 0
        finally:
            cap.release()

    @staticmethod
    def _resize_for_qwen(frame: Any, max_side: int = 960) -> Any:
        h, w = frame.shape[:2]
        if max(h, w) <= max_side:
            return frame
        scale = max_side / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
