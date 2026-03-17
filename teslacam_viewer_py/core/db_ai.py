from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from core.models import Event, VideoClip
except ModuleNotFoundError:
    from teslacam_viewer_py.core.models import Event, VideoClip

PRIMARY_TYPES = (
    "person_passby",
    "person_loitering",
    "person_close_approach",
    "vehicle_passby",
    "vehicle_parking_nearby",
    "vehicle_close_approach",
    "possible_impact",
    "confirmed_impact",
    "lighting_change",
    "night_activity",
    "animal_passby",
    "noise_false_trigger",
    "other",
)

PRIMARY_TYPE_LABELS = {
    "person_passby": "行人路过",
    "person_loitering": "行人徘徊",
    "person_close_approach": "行人靠近车辆",
    "vehicle_passby": "车辆路过",
    "vehicle_parking_nearby": "车辆停靠附近",
    "vehicle_close_approach": "车辆靠近",
    "possible_impact": "疑似撞击",
    "confirmed_impact": "确认撞击",
    "lighting_change": "光照突变",
    "night_activity": "夜间活动",
    "animal_passby": "动物经过",
    "noise_false_trigger": "噪声误报",
    "other": "其他",
}

AI_STATUS_LABELS = {
    "ok": "正常",
    "skipped": "已跳过",
    "error": "异常",
}

QWEN_TRIGGER_TYPES = {
    "person_loitering",
    "person_close_approach",
    "possible_impact",
    "vehicle_close_approach",
}


def primary_type_label(primary_type: str) -> str:
    return PRIMARY_TYPE_LABELS.get(primary_type, primary_type)


def ai_status_label(status: str) -> str:
    return AI_STATUS_LABELS.get(status, status)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_key_from_event(event: Event) -> str:
    ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    groups = ",".join(sorted(event.source_groups)) or "Other"
    return f"{ts}|{groups}"


def event_hash(event: Event, rule_version: str, model_version: str) -> str:
    parts = [event_key_from_event(event), rule_version, model_version]
    for cam, clip in sorted(event.clips.items(), key=lambda item: item[0]):
        try:
            stat = clip.path.stat()
            size = stat.st_size
            mtime = int(stat.st_mtime)
        except OSError:
            size = -1
            mtime = -1
        parts.append(f"{cam}|{clip.path.resolve()}|{size}|{mtime}")
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def normalize_roi(roi: dict[str, Any] | None) -> dict[str, float]:
    base = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    if not roi:
        return base
    try:
        x = float(roi.get("x", 0.0))
        y = float(roi.get("y", 0.0))
        w = float(roi.get("w", 1.0))
        h = float(roi.get("h", 1.0))
    except (TypeError, ValueError):
        return base
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.01), 1.0)
    h = min(max(h, 0.01), 1.0)
    if x + w > 1.0:
        x = max(0.0, 1.0 - w)
    if y + h > 1.0:
        y = max(0.0, 1.0 - h)
    return {"x": x, "y": y, "w": w, "h": h}


@dataclass(slots=True)
class AIResultRecord:
    event_key: str
    version: str
    primary_type: str
    secondary_types: list[str]
    risk_level: int
    objects: dict[str, int]
    ai_summary: str
    evidence: list[dict[str, Any]]
    analyzer_source: str
    status: str
    error_message: str
    event_hash: str
    updated_at: str = ""

    def with_defaults(self) -> "AIResultRecord":
        primary = self.primary_type if self.primary_type in PRIMARY_TYPES else "other"
        risk = max(0, min(3, int(self.risk_level)))
        return AIResultRecord(
            event_key=self.event_key,
            version=self.version,
            primary_type=primary,
            secondary_types=list(self.secondary_types or []),
            risk_level=risk,
            objects={
                "person": int((self.objects or {}).get("person", 0)),
                "vehicle": int((self.objects or {}).get("vehicle", 0)),
                "animal": int((self.objects or {}).get("animal", 0)),
            },
            ai_summary=str(self.ai_summary or "").strip(),
            evidence=list(self.evidence or []),
            analyzer_source=str(self.analyzer_source or "yolo+rules"),
            status=str(self.status or "ok"),
            error_message=str(self.error_message or ""),
            event_hash=str(self.event_hash or ""),
            updated_at=self.updated_at or utc_now_iso(),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AIResultRecord":
        return cls(
            event_key=row["event_key"],
            version=row["version"],
            primary_type=row["primary_type"],
            secondary_types=json.loads(row["secondary_types"] or "[]"),
            risk_level=int(row["risk_level"]),
            objects=json.loads(row["objects"] or "{}"),
            ai_summary=row["ai_summary"] or "",
            evidence=json.loads(row["evidence"] or "[]"),
            analyzer_source=row["analyzer_source"] or "yolo+rules",
            status=row["status"] or "ok",
            error_message=row["error_message"] or "",
            event_hash=row["event_hash"] or "",
            updated_at=row["updated_at"] or "",
        ).with_defaults()


class AIResultStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._migrate_legacy_tables()
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    cam TEXT NOT NULL,
                    start_ts TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    source_folder TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    version TEXT NOT NULL,
                    event_hash TEXT NOT NULL DEFAULT '',
                    primary_type TEXT NOT NULL,
                    secondary_types TEXT NOT NULL DEFAULT '[]',
                    risk_level INTEGER NOT NULL DEFAULT 0,
                    objects TEXT NOT NULL DEFAULT '{}',
                    ai_summary TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '[]',
                    analyzer_source TEXT NOT NULL DEFAULT 'yolo+rules',
                    status TEXT NOT NULL DEFAULT 'ok',
                    error_message TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    UNIQUE(event_id, version),
                    FOREIGN KEY(event_id) REFERENCES events(id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roi_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cam TEXT NOT NULL UNIQUE,
                    roi_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_key ON events(event_key)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_results_risk ON ai_results(risk_level)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_results_type ON ai_results(primary_type)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_path ON clips(path)")

    def _migrate_legacy_tables(self) -> None:
        # Legacy schema had ai_results(event_key, primary_label, ...); rebuild to new normalized schema.
        cols = self._table_columns("ai_results")
        if cols and ("primary_type" not in cols or "event_id" not in cols or "version" not in cols):
            legacy_name = f"ai_results_legacy_{int(time.time())}"
            with self._conn:
                self._conn.execute(f"ALTER TABLE ai_results RENAME TO {legacy_name}")

    def _table_columns(self, table_name: str) -> set[str]:
        try:
            rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        except sqlite3.DatabaseError:
            return set()
        return {str(row[1]) for row in rows}

    def upsert_clip(self, clip: VideoClip, duration_ms: int = 0) -> None:
        payload = (
            str(clip.path.resolve()),
            clip.camera,
            clip.timestamp.isoformat(),
            int(max(duration_ms, 0)),
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO clips(path, cam, start_ts, duration_ms)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    cam=excluded.cam,
                    start_ts=excluded.start_ts,
                    duration_ms=CASE
                        WHEN excluded.duration_ms > 0 THEN excluded.duration_ms
                        ELSE clips.duration_ms
                    END
                """,
                payload,
            )

    def upsert_event(self, event_key: str, source_folder: str) -> int:
        now = utc_now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO events(event_key, created_at, source_folder)
                VALUES(?, ?, ?)
                ON CONFLICT(event_key) DO UPDATE SET source_folder=excluded.source_folder
                """,
                (event_key, now, source_folder),
            )
            row = self._conn.execute(
                "SELECT id FROM events WHERE event_key = ?",
                (event_key,),
            ).fetchone()
        if not row:
            raise RuntimeError(f"failed to upsert event: {event_key}")
        return int(row["id"])

    def upsert_ai_result(self, event_id: int, record: AIResultRecord) -> None:
        data = record.with_defaults()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO ai_results(
                    event_id, version, event_hash, primary_type, secondary_types,
                    risk_level, objects, ai_summary, evidence, analyzer_source,
                    status, error_message, updated_at
                ) VALUES(
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                ON CONFLICT(event_id, version) DO UPDATE SET
                    event_hash=excluded.event_hash,
                    primary_type=excluded.primary_type,
                    secondary_types=excluded.secondary_types,
                    risk_level=excluded.risk_level,
                    objects=excluded.objects,
                    ai_summary=excluded.ai_summary,
                    evidence=excluded.evidence,
                    analyzer_source=excluded.analyzer_source,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    event_id,
                    data.version,
                    data.event_hash,
                    data.primary_type,
                    json.dumps(data.secondary_types, ensure_ascii=False),
                    data.risk_level,
                    json.dumps(data.objects, ensure_ascii=False),
                    data.ai_summary,
                    json.dumps(data.evidence, ensure_ascii=False),
                    data.analyzer_source,
                    data.status,
                    data.error_message,
                    data.updated_at,
                ),
            )

    def save_event_analysis(
        self,
        event: Event,
        version: str,
        source_folder: str,
        record: AIResultRecord,
        durations_ms: dict[str, int] | None = None,
    ) -> None:
        durations_ms = durations_ms or {}
        for clip in event.clips.values():
            duration = int(durations_ms.get(str(clip.path.resolve()), 0))
            self.upsert_clip(clip, duration_ms=duration)
        event_id = self.upsert_event(event_key_from_event(event), source_folder)
        self.upsert_ai_result(event_id, record)

    def get_result(self, event_key: str, version: str) -> AIResultRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT e.event_key, r.*
                FROM events e
                JOIN ai_results r ON r.event_id = e.id
                WHERE e.event_key = ? AND r.version = ?
                """,
                (event_key, version),
            ).fetchone()
        if not row:
            return None
        return AIResultRecord.from_row(row)

    def get_many(self, event_keys: Iterable[str], version: str) -> dict[str, AIResultRecord]:
        keys = [key for key in event_keys if key]
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        query = f"""
            SELECT e.event_key, r.*
            FROM events e
            JOIN ai_results r ON r.event_id = e.id
            WHERE e.event_key IN ({placeholders}) AND r.version = ?
        """
        params = [*keys, version]
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row["event_key"]: AIResultRecord.from_row(row) for row in rows}

    def get_many_latest(self, event_keys: Iterable[str]) -> dict[str, AIResultRecord]:
        keys = [key for key in event_keys if key]
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        query = f"""
            SELECT e.event_key, r.*
            FROM events e
            JOIN ai_results r ON r.event_id = e.id
            JOIN (
                SELECT event_id, MAX(updated_at) AS max_updated_at
                FROM ai_results
                GROUP BY event_id
            ) t ON t.event_id = r.event_id AND t.max_updated_at = r.updated_at
            WHERE e.event_key IN ({placeholders})
        """
        with self._lock:
            rows = self._conn.execute(query, keys).fetchall()
        return {row["event_key"]: AIResultRecord.from_row(row) for row in rows}

    def get_all_rois(self) -> dict[str, dict[str, float]]:
        with self._lock:
            rows = self._conn.execute("SELECT cam, roi_json FROM roi_configs").fetchall()
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            try:
                roi_raw = json.loads(row["roi_json"] or "{}")
            except json.JSONDecodeError:
                roi_raw = {}
            result[row["cam"]] = normalize_roi(roi_raw)
        return result

    def upsert_roi(self, cam: str, roi: dict[str, Any]) -> None:
        roi_norm = normalize_roi(roi)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO roi_configs(cam, roi_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(cam) DO UPDATE SET
                    roi_json=excluded.roi_json,
                    updated_at=excluded.updated_at
                """,
                (
                    cam,
                    json.dumps(roi_norm, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
