from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set

try:
    from core.models import CLIP_GROUPS, VideoClip
except ModuleNotFoundError:
    from teslacam_viewer_py.core.models import CLIP_GROUPS, VideoClip

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}
KNOWN_CLIP_DIRS = {"recentclips", "savedclips", "sentryclips"}
CLIP_DIR_TO_GROUP = {name.lower(): name for name in CLIP_GROUPS}

CANONICAL_CAMERA_ALIASES = {
    "front": "front",
    "fwd": "front",
    "forward": "front",
    "front_narrow": "front_narrow",
    "narrow_front": "front_narrow",
    "narrow": "front_narrow",
    "left_repeater": "left_repeater",
    "left-repeater": "left_repeater",
    "left": "left_repeater",
    "driver": "left_repeater",
    "right_repeater": "right_repeater",
    "right-repeater": "right_repeater",
    "right": "right_repeater",
    "passenger": "right_repeater",
    "rear": "back",
    "back": "back",
    "rearview": "back",
    "rear_view": "back",
    "left_pillar": "left_pillar",
    "right_pillar": "right_pillar",
    "pillar_left": "left_pillar",
    "pillar_right": "right_pillar",
    "left_b_pillar": "left_b_pillar",
    "right_b_pillar": "right_b_pillar",
    "b_pillar_left": "left_b_pillar",
    "b_pillar_right": "right_b_pillar",
}
COMMON_NON_CAMERA_TOKENS = {"clip", "video", "teslacam", "camera", "cam"}

TIMESTAMP_PATTERNS = (
    # YYYY-MM-DD_HH-MM-SS-<cam>
    re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})[_ ](?P<time>\d{2}-\d{2}-\d{2})-(?P<cam>[a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    ),
    # YYYY-MM-DD_HH-MM-SS_<cam>
    re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})[_ ](?P<time>\d{2}-\d{2}-\d{2})_(?P<cam>[a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    ),
    # YYYYMMDD_HHMMSS_<cam>
    re.compile(
        r"(?P<date_compact>\d{8})[_ ](?P<time_compact>\d{6})[_-](?P<cam>[a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    ),
    # Fallback: timestamp only (no camera token)
    re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})[_ ](?P<time>\d{2}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<date_compact>\d{8})[_ ](?P<time_compact>\d{6})",
        re.IGNORECASE,
    ),
)


def resolve_scan_roots(selected_dir: Path) -> Sequence[Path]:
    selected = selected_dir.expanduser().resolve()
    if not selected.exists() or not selected.is_dir():
        return []

    name_lower = selected.name.lower()
    if name_lower in KNOWN_CLIP_DIRS:
        return [selected]

    child_dirs = [p for p in selected.iterdir() if p.is_dir() and p.name.lower() in KNOWN_CLIP_DIRS]
    if child_dirs:
        return sorted(child_dirs, key=lambda p: p.name.lower())

    return [selected]


def scan_videos(selected_dir: Path) -> List[VideoClip]:
    roots = resolve_scan_roots(selected_dir)
    clips: List[VideoClip] = []
    for root in roots:
        for file_path in iter_video_files(root):
            clip = parse_clip(file_path)
            if clip is not None:
                clips.append(clip)
    return clips


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


DRIVING_STATE_KEYWORDS = {
    "emergency_brake": "紧急制动",
    "hard_brake": "紧急制动",
    "brake": "制动",
    "honk": "按喇叭",
    "horn": "按喇叭",
    "sentry": "哨兵",
    "collision": "碰撞",
    "impact": "碰撞",
    "rollover": "翻车",
    "park": "泊车",
    "away": "离开",
    "manual": "手动",
}


def parse_clip(file_path: Path) -> VideoClip | None:
    stem = file_path.stem
    timestamp, cam_hint = parse_timestamp_and_camera(stem)

    camera = normalize_camera_token(cam_hint) if cam_hint else infer_camera(stem)
    if not camera:
        return None

    mtime_fallback = False
    if timestamp is None:
        mtime_fallback = True
        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).replace(microsecond=0)

    flags = parse_driving_state_flags(file_path)

    return VideoClip(
        path=file_path,
        timestamp=timestamp,
        camera=camera,
        source_group=infer_source_group(file_path),
        flags=flags,
        mtime_fallback=mtime_fallback,
    )


def parse_driving_state_flags(file_path: Path) -> Set[str]:
    text = (file_path.name + " " + str(file_path.parent)).lower()
    found: Set[str] = set()
    for keyword, label in DRIVING_STATE_KEYWORDS.items():
        if keyword in text:
            found.add(label)
    return found


def parse_timestamp_and_camera(stem: str) -> tuple[datetime | None, str | None]:
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue

        groups = match.groupdict()
        timestamp = parse_groups_timestamp(groups)
        if timestamp is None:
            continue
        return timestamp, groups.get("cam")
    return None, None


def parse_groups_timestamp(groups: dict[str, str | None]) -> datetime | None:
    date_part = groups.get("date")
    time_part = groups.get("time")
    if date_part and time_part:
        try:
            return datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            return None

    date_compact = groups.get("date_compact")
    time_compact = groups.get("time_compact")
    if date_compact and time_compact:
        try:
            return datetime.strptime(f"{date_compact}_{time_compact}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def infer_camera(text: str) -> str | None:
    normalized = text.lower().replace("-", "_")
    tokens = [tok for tok in re.split(r"[^a-z0-9_]+", normalized) if tok]

    for token in reversed(tokens):
        camera = normalize_camera_token(token, allow_unknown=False)
        if camera:
            return camera

    for alias, canonical in sorted(CANONICAL_CAMERA_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias.replace("-", "_") in normalized:
            return canonical

    for token in reversed(tokens):
        if token in COMMON_NON_CAMERA_TOKENS:
            continue
        if token.isdigit():
            continue
        if not re.search(r"[a-z]", token):
            continue
        return normalize_camera_token(token, allow_unknown=True)

    return None


def normalize_camera_token(token: str | None, allow_unknown: bool = True) -> str | None:
    if not token:
        return None
    normalized = token.lower().strip("_- ").replace("-", "_")
    if not normalized:
        return None

    canonical = CANONICAL_CAMERA_ALIASES.get(normalized)
    if canonical:
        return canonical

    if normalized.endswith("_camera"):
        normalized = normalized[: -len("_camera")]
        canonical = CANONICAL_CAMERA_ALIASES.get(normalized)
        if canonical:
            return canonical

    if not allow_unknown:
        return None
    if not re.search(r"[a-z]", normalized):
        return None
    return normalized


def infer_source_group(file_path: Path) -> str:
    for part in file_path.parts:
        group = CLIP_DIR_TO_GROUP.get(part.lower())
        if group:
            return group
    return "Other"
