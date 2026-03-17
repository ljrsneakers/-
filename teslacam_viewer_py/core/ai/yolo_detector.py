from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import config
except ModuleNotFoundError:
    from teslacam_viewer_py import config


@dataclass(slots=True)
class Detection:
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class YoloDetector:
    _vehicle_classes = {1, 2, 3, 5, 7}
    _animal_classes = {14, 15, 16, 17, 18, 19, 21}

    def __init__(self, model_name: str | None = None, conf_threshold: float | None = None) -> None:
        self.model_name = model_name or config.AI_YOLO_MODEL
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.AI_DET_CONF
        self._model = None
        self.error_message = ""
        self._load_model()

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        if not config.AI_ENABLE_YOLO:
            self.error_message = "YOLO disabled by config"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_name)
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self.error_message = f"load model failed: {exc}"

    def detect(self, frame: Any) -> list[Detection]:
        if self._model is None:
            return []
        try:
            result = self._model(frame, verbose=False, imgsz=640)[0]
        except Exception:
            return []
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            mapped = self._map_class(cls)
            if mapped is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=mapped,
                    conf=conf,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return detections

    def _map_class(self, class_id: int) -> str | None:
        if class_id == 0:
            return "person"
        if class_id in self._vehicle_classes:
            return "vehicle"
        if class_id in self._animal_classes:
            return "animal"
        return None
