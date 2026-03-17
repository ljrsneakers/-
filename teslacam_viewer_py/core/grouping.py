from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

try:
    from core.models import Event, GroupByMode, VideoClip
except ModuleNotFoundError:
    from teslacam_viewer_py.core.models import Event, GroupByMode, VideoClip


def group_events(clips: Iterable[VideoClip], group_by: GroupByMode = "second") -> List[Event]:
    bucket: Dict[datetime, Dict[str, VideoClip]] = defaultdict(dict)
    source_bucket: Dict[datetime, set[str]] = defaultdict(set)
    flag_bucket: Dict[datetime, set[str]] = defaultdict(set)
    for clip in clips:
        event_key = make_group_key(clip.timestamp, group_by)
        source_bucket[event_key].add(clip.source_group)
        flag_bucket[event_key].update(clip.flags)
        current = bucket[event_key].get(clip.camera)
        if current is None:
            bucket[event_key][clip.camera] = clip
            continue
        # Keep the larger file when duplicate camera clips hit same second.
        if clip.path.stat().st_size >= current.path.stat().st_size:
            bucket[event_key][clip.camera] = clip

    events = [
        Event(
            timestamp=ts,
            clips=cams,
            source_groups=source_bucket.get(ts, set()),
            flags=flag_bucket.get(ts, set()),
        )
        for ts, cams in bucket.items()
    ]
    events.sort(key=lambda event: event.timestamp, reverse=True)
    return events


def make_group_key(timestamp: datetime, group_by: GroupByMode) -> datetime:
    mode = group_by.lower()
    if mode == "minute":
        return timestamp.replace(second=0, microsecond=0)
    return timestamp.replace(microsecond=0)
