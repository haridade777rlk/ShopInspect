"""YOLO detector wrapper -> unified result dict."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import Settings, get_settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def resize_for_infer(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale long side to max_side. Returns (image, scale) where scale maps new->orig (usually <=1)."""
    if max_side <= 0:
        return frame_bgr, 1.0
    h, w = frame_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return frame_bgr, 1.0
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


@dataclass
class DetectResult:
    created_at: str
    source: str
    detections: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    annotated_bgr: np.ndarray | None = None
    image_path: str | None = None
    note: str | None = None
    elapsed_ms: float = 0.0
    conf_used: float | None = None
    image_width: int | None = None
    image_height: int | None = None

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def to_record_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "source": self.source,
            "image_path": self.image_path,
            "num_detections": self.num_detections,
            "detections": self.detections,
            "model": self.model,
            "note": self.note,
        }


class Detector:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self.model_name = self.settings.model_path
        self.loaded = False
        self.load_error: str | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.load_error = f"ultralytics not installed: {e}"
            raise RuntimeError(self.load_error) from e

        model_path = self.settings.resolved_model_path()
        self.model_name = Path(model_path).name
        try:
            self._model = YOLO(model_path)
            self.loaded = True
            self.load_error = None
        except Exception as e:  # noqa: BLE001
            self.load_error = str(e)
            self.loaded = False
            raise RuntimeError(f"failed to load model '{model_path}': {e}") from e

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def predict_bgr(
        self,
        frame_bgr: np.ndarray,
        source: str = "file",
        note: str | None = None,
        conf: float | None = None,
    ) -> DetectResult:
        self.ensure_loaded()
        assert self._model is not None

        conf_used = float(self.settings.confidence if conf is None else conf)
        infer_img, scale = resize_for_infer(frame_bgr, int(self.settings.max_infer_side))

        t0 = time.perf_counter()
        results = self._model.predict(
            source=infer_img,
            conf=conf_used,
            iou=self.settings.iou,
            device=self.settings.device,
            imgsz=self.settings.imgsz,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        r0 = results[0]
        detections: list[dict[str, Any]] = []
        names = r0.names or {}
        # scale boxes back to original frame coords if downscaled
        inv = (1.0 / scale) if scale > 0 else 1.0
        if r0.boxes is not None and len(r0.boxes):
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, confs, clss):
                label = names.get(int(cls_id), str(int(cls_id)))
                x1, y1, x2, y2 = [float(v) * inv for v in box.tolist()]
                detections.append(
                    {
                        "label": str(label),
                        "confidence": float(round(float(score), 4)),
                        "bbox_xyxy": [float(round(x1, 2)), float(round(y1, 2)), float(round(x2, 2)), float(round(y2, 2))],
                    }
                )

        # draw on original-sized frame for nicer saved previews
        annotated = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in d["bbox_xyxy"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (59, 130, 246), 2)
            tag = f"{d['label']} {d['confidence']:.2f}"
            cv2.putText(
                annotated,
                tag,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (34, 197, 94),
                2,
            )

        h0, w0 = frame_bgr.shape[:2]
        return DetectResult(
            created_at=_utc_now_iso(),
            source=source,
            detections=detections,
            model=self.model_name,
            annotated_bgr=annotated,
            note=note,
            elapsed_ms=elapsed_ms,
            conf_used=conf_used,
            image_width=int(w0),
            image_height=int(h0),
        )

    def predict_path(
        self,
        image_path: str | Path,
        source: str = "path",
        note: str | None = None,
        conf: float | None = None,
    ) -> DetectResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"cannot read image: {path}")
        return self.predict_bgr(frame, source=source, note=note, conf=conf)

    def save_annotated(self, result: DetectResult, prefix: str = "det") -> Path:
        if result.annotated_bgr is None:
            raise ValueError("no annotated image to save")
        out_dir = self.settings.outputs_path
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"{prefix}_{ts}.jpg"
        quality = int(getattr(self.settings, "jpeg_quality", 85) or 85)
        ok = cv2.imwrite(
            str(out_path),
            result.annotated_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, quality))],
        )
        if not ok:
            raise RuntimeError(f"failed to write {out_path}")
        try:
            rel = out_path.resolve().relative_to(self.settings.project_root)
            result.image_path = str(rel).replace("\\", "/")
        except ValueError:
            result.image_path = str(out_path)
        return out_path


_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector
