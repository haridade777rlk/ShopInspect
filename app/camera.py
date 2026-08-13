"""Camera open helpers for Windows."""
from __future__ import annotations

import cv2

from app.config import Settings, get_settings


def open_camera(settings: Settings | None = None) -> cv2.VideoCapture:
    settings = settings or get_settings()
    index = int(settings.camera_index)
    backend = (settings.camera_backend or "dshow").lower()
    if backend == "dshow":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        # fallback once
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"cannot open camera index={index}. "
            "Close Teams/WeChat/Camera app; check Windows Privacy > Camera."
        )
    return cap


def probe_cameras(max_index: int = 5, backend: str = "dshow") -> list[dict]:
    results = []
    for i in range(max_index):
        if backend == "dshow":
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
        opened = cap.isOpened()
        shape = None
        read_ok = False
        if opened:
            read_ok, frame = cap.read()
            if read_ok and frame is not None:
                shape = list(frame.shape)
        cap.release()
        results.append(
            {
                "index": i,
                "opened": opened,
                "read_ok": read_ok,
                "shape": shape,
            }
        )
    return results
