from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None

try:
    import config
    from core.db_ai import PRIMARY_TYPES
except ModuleNotFoundError:
    from teslacam_viewer_py import config
    from teslacam_viewer_py.core.db_ai import PRIMARY_TYPES

logger = logging.getLogger(__name__)


class QwenClientError(RuntimeError):
    pass


@dataclass(slots=True)
class QwenResult:
    primary_type: str
    secondary_types: list[str]
    risk_level: int
    ai_summary: str
    objects: dict[str, int]
    notes: str
    evidence: list[dict[str, Any]]


class QwenVLClient:
    def __init__(
        self,
        endpoint: str | None = None,
        model_name: str | None = None,
        timeout_sec: int | None = None,
        retries: int | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout_sec = timeout_sec or config.AI_TIMEOUT_SEC
        self.retries = retries if retries is not None else config.AI_HTTP_RETRIES

    def available(self) -> bool:
        return bool(config.get_dashscope_api_key())

    def analyze_images(
        self,
        image_paths: list[Path],
        local_hint: dict[str, Any] | None = None,
        force_retry: bool = False,
    ) -> tuple[QwenResult, str]:
        if requests is None:
            raise QwenClientError("缺少 requests 依赖")
        if not self.available():
            raise QwenClientError("未配置 DASHSCOPE_API_KEY")
        if not image_paths:
            raise QwenClientError("没有可分析图片")

        endpoint = (self.endpoint or config.get_dashscope_endpoint()).strip()
        model = (self.model_name or config.get_qwen_model_name()).strip()
        api_key = config.get_dashscope_api_key().strip()
        prompt = self._build_prompt(local_hint or {}, strict=force_retry)

        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": self._to_data_url(path)}})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是 TeslaCam 安全分析助手，必须只返回 JSON 对象。"},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        last_error = ""
        total = max(1, self.retries + 1)
        for idx in range(total):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
                response.raise_for_status()
                raw = response.text
                parsed = self._parse_response(response.json())
                return parsed, raw
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("Qwen 请求失败 (%d/%d): %s", idx + 1, total, exc)
        raise QwenClientError(last_error or "Qwen 请求失败")

    def _build_prompt(self, local_hint: dict[str, Any], strict: bool) -> str:
        type_text = " / ".join(PRIMARY_TYPES)
        strict_note = (
            "注意：绝对不能输出 markdown、代码块、解释文本，只允许一个 JSON 对象。"
            if strict
            else "只输出 JSON 对象。"
        )
        return (
            "你将收到多路 TeslaCam 关键帧，请做事件分类。\n"
            f"primary_type 必须是以下枚举之一：{type_text}\n"
            "risk_level 范围 0-3。\n"
            "输出 JSON schema:\n"
            "{\n"
            '  "primary_type":"...",\n'
            '  "secondary_types":["..."],\n'
            '  "risk_level":0,\n'
            '  "ai_summary":"一句话中文摘要",\n'
            '  "objects":{"person":0,"vehicle":0,"animal":0},\n'
            '  "notes":"optional",\n'
            '  "evidence":[{"cam":"front","t_ms":1234,"why":"closest approach"}]\n'
            "}\n"
            f"本地规则参考（仅参考，不必照搬）：{json.dumps(local_hint, ensure_ascii=False)}\n"
            f"{strict_note}"
        )

    def _to_data_url(self, image_path: Path) -> str:
        with image_path.open("rb") as file:
            data = base64.b64encode(file.read()).decode("utf-8")
        ext = image_path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
        return f"data:image/{mime};base64,{data}"

    def _parse_response(self, payload: dict[str, Any]) -> QwenResult:
        try:
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise QwenClientError(f"返回结构异常: {exc}") from exc
        text = ""
        if isinstance(content, list):
            text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        else:
            text = str(content)
        obj = self._safe_json_load(text)
        return self._validate_schema(obj)

    def _safe_json_load(self, raw_text: str) -> dict[str, Any]:
        raw = raw_text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise QwenClientError("无法解析 Qwen 返回的 JSON")

    def _validate_schema(self, data: dict[str, Any]) -> QwenResult:
        primary_type = str(data.get("primary_type", "other")).strip()
        if primary_type not in PRIMARY_TYPES:
            raise QwenClientError(f"primary_type 非法: {primary_type}")

        risk_level = int(data.get("risk_level", 0))
        if risk_level < 0 or risk_level > 3:
            raise QwenClientError(f"risk_level 非法: {risk_level}")

        secondary = data.get("secondary_types", []) or []
        if not isinstance(secondary, list):
            secondary = []
        secondary_types = [str(item).strip() for item in secondary if str(item).strip()]
        secondary_types = [item for item in secondary_types if item in PRIMARY_TYPES and item != primary_type]

        objects = data.get("objects", {}) or {}
        person = max(0, int(objects.get("person", 0)))
        vehicle = max(0, int(objects.get("vehicle", 0)))
        animal = max(0, int(objects.get("animal", 0)))

        evidence_raw = data.get("evidence", []) or []
        evidence: list[dict[str, Any]] = []
        if isinstance(evidence_raw, list):
            for item in evidence_raw[:12]:
                if not isinstance(item, dict):
                    continue
                evidence.append(
                    {
                        "cam": str(item.get("cam", "")).strip() or "unknown",
                        "t_ms": max(0, int(item.get("t_ms", 0))),
                        "why": str(item.get("why", "")).strip(),
                    }
                )

        return QwenResult(
            primary_type=primary_type,
            secondary_types=secondary_types,
            risk_level=risk_level,
            ai_summary=str(data.get("ai_summary", "")).strip(),
            objects={"person": person, "vehicle": vehicle, "animal": animal},
            notes=str(data.get("notes", "")).strip(),
            evidence=evidence,
        )
