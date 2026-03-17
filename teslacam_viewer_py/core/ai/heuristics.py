from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import config
    from core.db_ai import PRIMARY_TYPES, QWEN_TRIGGER_TYPES, normalize_roi
except ModuleNotFoundError:
    from teslacam_viewer_py import config
    from teslacam_viewer_py.core.db_ai import PRIMARY_TYPES, QWEN_TRIGGER_TYPES, normalize_roi


@dataclass(slots=True)
class FrameSignal:
    cam: str
    t_ms: int
    brightness: float
    diff_energy: float
    person_count: int
    vehicle_count: int
    animal_count: int
    person_max_area_ratio: float
    vehicle_max_area_ratio: float
    person_in_roi: bool
    vehicle_in_roi: bool


@dataclass(slots=True)
class HeuristicResult:
    primary_type: str
    secondary_types: list[str]
    risk_level: int
    ai_summary: str
    objects: dict[str, int]
    evidence: list[dict[str, Any]]
    metrics: dict[str, Any]
    qwen_candidate: bool


def classify_event(
    event_ts: datetime,
    signals: list[FrameSignal],
    sampling_fps: float,
) -> HeuristicResult:
    if not signals:
        return HeuristicResult(
            primary_type="other",
            secondary_types=[],
            risk_level=0,
            ai_summary="未检测到有效帧，归类为其他事件。",
            objects={"person": 0, "vehicle": 0, "animal": 0},
            evidence=[],
            metrics={"frames": 0},
            qwen_candidate=False,
        )

    fps = max(0.1, float(sampling_fps))
    person_frames = sum(1 for s in signals if s.person_count > 0)
    vehicle_frames = sum(1 for s in signals if s.vehicle_count > 0)
    animal_frames = sum(1 for s in signals if s.animal_count > 0)
    person_roi_frames = sum(1 for s in signals if s.person_in_roi)
    vehicle_roi_frames = sum(1 for s in signals if s.vehicle_in_roi)
    person_seconds = person_frames / fps
    vehicle_seconds = vehicle_frames / fps

    max_person_count = max(s.person_count for s in signals)
    max_vehicle_count = max(s.vehicle_count for s in signals)
    max_animal_count = max(s.animal_count for s in signals)

    person_areas = [s.person_max_area_ratio for s in signals if s.person_count > 0]
    vehicle_areas = [s.vehicle_max_area_ratio for s in signals if s.vehicle_count > 0]
    person_growth = (max(person_areas) - min(person_areas)) if len(person_areas) >= 2 else 0.0
    vehicle_growth = (max(vehicle_areas) - min(vehicle_areas)) if len(vehicle_areas) >= 2 else 0.0

    brightness_values = [s.brightness for s in signals]
    diff_values = [s.diff_energy for s in signals]
    max_brightness_jump = _max_delta(brightness_values)
    diff_peak = max(diff_values) if diff_values else 0.0

    person_close = person_roi_frames > 0 or person_growth >= config.AI_CLOSE_AREA_GROWTH
    vehicle_close = vehicle_roi_frames > 0 or vehicle_growth >= config.AI_CLOSE_AREA_GROWTH
    lighting_change = max_brightness_jump >= config.AI_LIGHTING_CHANGE_THRESHOLD
    possible_impact = diff_peak >= config.AI_IMPACT_DIFF_THRESHOLD and (person_close or vehicle_close)
    confirmed_impact = diff_peak >= (config.AI_IMPACT_DIFF_THRESHOLD * 1.6) and (
        person_roi_frames >= 2 or vehicle_roi_frames >= 2
    )
    night_activity = event_ts.hour >= 22 or event_ts.hour < 6
    has_objects = (person_frames + vehicle_frames + animal_frames) > 0
    noise_false_trigger = (not has_objects) and (lighting_change or diff_peak >= 8.0)

    person_loitering = person_roi_frames / fps >= config.AI_LOITER_SECONDS
    vehicle_parking = vehicle_roi_frames / fps >= config.AI_LOITER_SECONDS
    person_passby = person_frames >= config.AI_MIN_OBJECT_FRAMES and person_seconds < config.AI_PASSBY_MAX_SECONDS
    vehicle_passby = vehicle_frames >= config.AI_MIN_OBJECT_FRAMES and vehicle_seconds < config.AI_PASSBY_MAX_SECONDS

    secondary: list[str] = []
    if night_activity and (person_frames > 0 or vehicle_frames > 0):
        secondary.append("night_activity")
    if lighting_change:
        secondary.append("lighting_change")
    if animal_frames > 0:
        secondary.append("animal_passby")

    primary = "other"
    risk = 0
    if confirmed_impact:
        primary, risk = "confirmed_impact", 3
    elif possible_impact:
        primary, risk = "possible_impact", 3
    elif person_loitering:
        primary, risk = "person_loitering", 2
    elif person_close:
        primary, risk = "person_close_approach", 2
    elif vehicle_close:
        primary, risk = "vehicle_close_approach", 2
    elif vehicle_parking:
        primary, risk = "vehicle_parking_nearby", 1
    elif person_passby:
        primary, risk = "person_passby", 1
    elif vehicle_passby:
        primary, risk = "vehicle_passby", 1
    elif animal_frames > 0:
        primary, risk = "animal_passby", 1
    elif lighting_change:
        primary, risk = "lighting_change", 1
    elif noise_false_trigger:
        primary, risk = "noise_false_trigger", 0
    elif night_activity and has_objects:
        primary, risk = "night_activity", 1

    if primary not in PRIMARY_TYPES:
        primary = "other"
        risk = 0

    if primary not in secondary and primary not in {"other"}:
        pass
    secondary = [s for s in secondary if s != primary]

    evidence = _build_evidence(signals, lighting_change, possible_impact, person_close, vehicle_close)
    objects = {"person": int(max_person_count), "vehicle": int(max_vehicle_count), "animal": int(max_animal_count)}
    summary = _build_summary(primary, objects, risk, night_activity, lighting_change)
    qwen_candidate = risk >= config.AI_QWEN_TRIGGER_MIN_RISK or primary in QWEN_TRIGGER_TYPES

    metrics = {
        "frames": len(signals),
        "person_frames": person_frames,
        "vehicle_frames": vehicle_frames,
        "animal_frames": animal_frames,
        "person_roi_frames": person_roi_frames,
        "vehicle_roi_frames": vehicle_roi_frames,
        "person_area_growth": round(person_growth, 4),
        "vehicle_area_growth": round(vehicle_growth, 4),
        "max_brightness_jump": round(max_brightness_jump, 2),
        "diff_peak": round(diff_peak, 2),
    }
    return HeuristicResult(
        primary_type=primary,
        secondary_types=secondary,
        risk_level=risk,
        ai_summary=summary,
        objects=objects,
        evidence=evidence,
        metrics=metrics,
        qwen_candidate=qwen_candidate,
    )


def point_in_roi(x: float, y: float, roi: dict[str, Any] | None) -> bool:
    norm = normalize_roi(roi)
    return norm["x"] <= x <= (norm["x"] + norm["w"]) and norm["y"] <= y <= (norm["y"] + norm["h"])


def box_center_in_roi(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
    roi: dict[str, Any] | None,
) -> bool:
    if frame_w <= 0 or frame_h <= 0:
        return False
    cx = ((x1 + x2) / 2.0) / float(frame_w)
    cy = ((y1 + y2) / 2.0) / float(frame_h)
    return point_in_roi(cx, cy, roi)


def _max_delta(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    max_delta = 0.0
    last = values[0]
    for val in values[1:]:
        max_delta = max(max_delta, abs(val - last))
        last = val
    return max_delta


def _pick_peak(signals: list[FrameSignal], key: str, why: str) -> dict[str, Any] | None:
    if not signals:
        return None
    best = max(signals, key=lambda item: getattr(item, key, 0.0))
    value = float(getattr(best, key, 0.0))
    if value <= 0:
        return None
    return {"cam": best.cam, "t_ms": best.t_ms, "why": why}


def _build_evidence(
    signals: list[FrameSignal],
    lighting_change: bool,
    possible_impact: bool,
    person_close: bool,
    vehicle_close: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if person_close:
        item = _pick_peak(signals, "person_max_area_ratio", "closest person")
        if item:
            items.append(item)
    if vehicle_close:
        item = _pick_peak(signals, "vehicle_max_area_ratio", "closest vehicle")
        if item:
            items.append(item)
    if possible_impact:
        item = _pick_peak(signals, "diff_energy", "impact-like motion spike")
        if item:
            items.append(item)
    if lighting_change:
        item = _pick_peak(signals, "brightness", "brightness change")
        if item:
            items.append(item)

    seen: set[tuple[str, int]] = set()
    dedup: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("cam", "")), int(item.get("t_ms", 0)))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup[:8]


def _build_summary(
    primary: str,
    objects: dict[str, int],
    risk: int,
    night_activity: bool,
    lighting_change: bool,
) -> str:
    snippets = [f"主类型={primary}", f"风险={risk}"]
    if objects.get("person", 0) > 0:
        snippets.append(f"行人最多{objects['person']}个")
    if objects.get("vehicle", 0) > 0:
        snippets.append(f"车辆最多{objects['vehicle']}个")
    if objects.get("animal", 0) > 0:
        snippets.append(f"动物最多{objects['animal']}个")
    if night_activity:
        snippets.append("夜间时段")
    if lighting_change:
        snippets.append("存在光照突变")
    return "，".join(snippets) + "。"
