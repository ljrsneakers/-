from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
APP_CONFIG_PATH = Path(
    os.getenv("TESLACAM_CONFIG_PATH", str(APP_DIR / "user_settings.json"))
).resolve()


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _load_settings_file() -> dict[str, Any]:
    if not APP_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


_USER_SETTINGS = _load_settings_file()

# DashScope / Qwen-VL
ENV_DASHSCOPE_API_KEY = _env_text("DASHSCOPE_API_KEY", "")
ENV_DASHSCOPE_ENDPOINT = _env_text(
    "DASHSCOPE_ENDPOINT",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
ENV_DASHSCOPE_INTL_ENDPOINT = _env_text(
    "DASHSCOPE_INTL_ENDPOINT",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
)
ENV_QWEN_MODEL_NAME = _env_text("QWEN_MODEL_NAME", "qwen-vl-max-latest")

DASHSCOPE_API_KEY = str(_USER_SETTINGS.get("dashscope_api_key", "")).strip() or ENV_DASHSCOPE_API_KEY
DASHSCOPE_ENDPOINT = str(_USER_SETTINGS.get("dashscope_endpoint", "")).strip() or ENV_DASHSCOPE_ENDPOINT
DASHSCOPE_INTL_ENDPOINT = ENV_DASHSCOPE_INTL_ENDPOINT
QWEN_MODEL_NAME = str(_USER_SETTINGS.get("qwen_model_name", "")).strip() or ENV_QWEN_MODEL_NAME

# Versions
AI_RULE_VERSION = _env_text("AI_RULE_VERSION", "rules_v2")
AI_ANALYZER_VERSION = _env_text("AI_ANALYZER_VERSION", "analyzer_v2")

# AI runtime
AI_TIMEOUT_SEC = _env_int("AI_TIMEOUT_SEC", 45)
AI_HTTP_RETRIES = _env_int("AI_HTTP_RETRIES", 2)
AI_JSON_RETRY = _env_int("AI_JSON_RETRY", 1)
AI_MAX_IMAGES = _env_int("AI_MAX_IMAGES", 12)
AI_SAMPLING_FPS = _env_float("AI_SAMPLING_FPS", 2.0)
AI_WINDOW_SECONDS = _env_int("AI_WINDOW_SECONDS", 10)

# Local CV / heuristic thresholds
AI_ENABLE_YOLO = _env_text("AI_ENABLE_YOLO", "1").lower() not in {"0", "false", "no"}
AI_YOLO_MODEL = _env_text("AI_YOLO_MODEL", "yolov8n.pt")
AI_DET_CONF = _env_float("AI_DET_CONF", 0.25)
AI_PASSBY_MAX_SECONDS = _env_int("AI_PASSBY_MAX_SECONDS", 5)
AI_LOITER_SECONDS = _env_int("AI_LOITER_SECONDS", 6)
AI_CLOSE_AREA_GROWTH = _env_float("AI_CLOSE_AREA_GROWTH", 0.08)
AI_LIGHTING_CHANGE_THRESHOLD = _env_float("AI_LIGHTING_CHANGE_THRESHOLD", 30.0)
AI_IMPACT_DIFF_THRESHOLD = _env_float("AI_IMPACT_DIFF_THRESHOLD", 35.0)
AI_MIN_OBJECT_FRAMES = _env_int("AI_MIN_OBJECT_FRAMES", 2)

# Qwen trigger
AI_QWEN_TRIGGER_MIN_RISK = _env_int("AI_QWEN_TRIGGER_MIN_RISK", 2)

# Storage
AI_DB_PATH = Path(_env_text("AI_DB_PATH", str(APP_DIR / "ai_results.db"))).resolve()
AI_FRAME_CACHE_DIR = Path(_env_text("AI_FRAME_CACHE_DIR", str(APP_DIR / ".ai_frames"))).resolve()


def get_dashscope_api_key() -> str:
    return DASHSCOPE_API_KEY


def get_dashscope_endpoint() -> str:
    return DASHSCOPE_ENDPOINT


def get_qwen_model_name() -> str:
    return QWEN_MODEL_NAME


def get_user_settings(mask_key: bool = False) -> dict[str, str]:
    key = get_dashscope_api_key()
    if mask_key and key:
        key = f"{key[:4]}***{key[-4:]}" if len(key) > 8 else "***"
    return {
        "dashscope_api_key": key,
        "dashscope_endpoint": get_dashscope_endpoint(),
        "qwen_model_name": get_qwen_model_name(),
    }


def save_user_settings(api_key: str, endpoint: str, model_name: str) -> dict[str, str]:
    global _USER_SETTINGS
    global DASHSCOPE_API_KEY
    global DASHSCOPE_ENDPOINT
    global QWEN_MODEL_NAME

    normalized = {
        "dashscope_api_key": api_key.strip(),
        "dashscope_endpoint": endpoint.strip() or ENV_DASHSCOPE_ENDPOINT,
        "qwen_model_name": model_name.strip() or ENV_QWEN_MODEL_NAME,
    }
    _USER_SETTINGS = normalized
    APP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_CONFIG_PATH.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    DASHSCOPE_API_KEY = normalized["dashscope_api_key"] or ENV_DASHSCOPE_API_KEY
    DASHSCOPE_ENDPOINT = normalized["dashscope_endpoint"] or ENV_DASHSCOPE_ENDPOINT
    QWEN_MODEL_NAME = normalized["qwen_model_name"] or ENV_QWEN_MODEL_NAME
    return get_user_settings(mask_key=False)
