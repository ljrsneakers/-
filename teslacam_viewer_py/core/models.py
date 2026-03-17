from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Set

CAMERAS = ("front", "left_repeater", "right_repeater", "back")
CAMERA_PRIORITY = (
    "front",
    "front_narrow",
    "left_repeater",
    "right_repeater",
    "back",
    "left_pillar",
    "right_pillar",
    "left_b_pillar",
    "right_b_pillar",
)
CLIP_GROUPS = ("RecentClips", "SavedClips", "SentryClips")
GroupByMode = Literal["second", "minute"]


@dataclass(slots=True)
class VideoClip:
    path: Path
    timestamp: datetime
    camera: str
    source_group: str = "Other"
    flags: Set[str] = field(default_factory=set)
    mtime_fallback: bool = False


@dataclass(slots=True)
class Event:
    timestamp: datetime
    clips: Dict[str, VideoClip] = field(default_factory=dict)
    source_groups: Set[str] = field(default_factory=set)
    flags: Set[str] = field(default_factory=set)

    def clip_for(self, camera: str) -> Optional[VideoClip]:
        return self.clips.get(camera)

    def all_cameras(self) -> list[str]:
        return sorted(self.clips.keys())
