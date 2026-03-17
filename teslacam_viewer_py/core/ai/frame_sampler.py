from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:
    import config
except ModuleNotFoundError:
    from teslacam_viewer_py import config


@dataclass(slots=True)
class FrameCandidate:
    cam: str
    t_ms: int
    frame: Any
    score: float
    why: str


@dataclass(slots=True)
class SampledImage:
    cam: str
    t_ms: int
    path: Path
    score: float
    why: str


class EventFrameSampler:
    def __init__(self) -> None:
        self.frame_root = config.AI_FRAME_CACHE_DIR
        self.frame_root.mkdir(parents=True, exist_ok=True)

    def sample_for_qwen(
        self,
        candidates: list[FrameCandidate],
        max_images: int,
    ) -> list[SampledImage]:
        if cv2 is None or not candidates:
            return []
        limit = max(1, max_images)
        temp_dir = Path(tempfile.mkdtemp(prefix="qwen_", dir=self.frame_root))
        selected = self._select_candidates(candidates, limit)

        images: list[SampledImage] = []
        for idx, cand in enumerate(selected):
            filename = f"{cand.cam}_{cand.t_ms:06d}_{idx:02d}.jpg"
            path = temp_dir / filename
            cv2.imwrite(str(path), cand.frame)
            images.append(
                SampledImage(
                    cam=cand.cam,
                    t_ms=cand.t_ms,
                    path=path,
                    score=cand.score,
                    why=cand.why,
                )
            )
        return images

    def cleanup(self, images: list[SampledImage]) -> None:
        if not images:
            return
        folder = images[0].path.parent
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    def _select_candidates(self, candidates: list[FrameCandidate], max_images: int) -> list[FrameCandidate]:
        # Priority: close approach / impact spikes first.
        priority_order = {"closest person": 0, "closest vehicle": 1, "impact-like motion spike": 2}
        ordered = sorted(
            candidates,
            key=lambda c: (priority_order.get(c.why, 9), -c.score, c.t_ms),
        )

        selected: list[FrameCandidate] = []
        per_cam_count: dict[str, int] = {}
        for cand in ordered:
            if len(selected) >= max_images:
                break
            if any(c.cam == cand.cam and abs(c.t_ms - cand.t_ms) < 900 for c in selected):
                continue
            if per_cam_count.get(cand.cam, 0) >= max(2, max_images // 3):
                continue
            selected.append(cand)
            per_cam_count[cand.cam] = per_cam_count.get(cand.cam, 0) + 1

        if len(selected) < max_images:
            fallback = sorted(candidates, key=lambda c: (-c.score, c.t_ms))
            for cand in fallback:
                if len(selected) >= max_images:
                    break
                if any(c.cam == cand.cam and abs(c.t_ms - cand.t_ms) < 700 for c in selected):
                    continue
                selected.append(cand)
        return selected[:max_images]


def iter_sampled_frames(video_path: Path, sampling_fps: float, max_seconds: int):
    if cv2 is None:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(0.1, sampling_fps))))
    max_frames = int(max_seconds * fps)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index > max_frames:
                break
            if index % step == 0:
                t_ms = int((index / fps) * 1000)
                yield t_ms, frame
            index += 1
    finally:
        cap.release()


def frame_diff_energy(prev_gray: Any, gray: Any) -> float:
    if cv2 is None or prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    return float(diff.mean())
