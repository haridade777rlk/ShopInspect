"""Desktop live camera + YOLO preview.
Keys: q quit | s save annotated frame to data/outputs
Run: python scripts/run_cam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from app.camera import open_camera
from app.config import load_settings
from app.db import init_db, insert_record
from app.detector import Detector


def main() -> None:
    settings = load_settings()
    init_db(settings)
    print(f"[{settings.app_name_en}] model={settings.model_path} device={settings.device}")
    print("Loading model (first time may download weights)...")
    det = Detector(settings)
    det.load()
    print("Opening camera...")
    try:
        cap = open_camera(settings)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    win = f"{settings.app_name} - press q quit, s save"
    print(win)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: failed to read frame")
                break
            result = det.predict_bgr(frame, source="camera")
            shown = result.annotated_bgr if result.annotated_bgr is not None else frame
            # HUD
            hud = shown.copy()
            text = f"n={result.num_detections}  {result.elapsed_ms:.0f}ms  model={result.model}"
            cv2.putText(
                hud, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow(win, hud)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = det.save_annotated(result, prefix="camera")
                rid = insert_record(
                    created_at=result.created_at,
                    source="camera",
                    image_path=result.image_path,
                    num_detections=result.num_detections,
                    detections=result.detections,
                    model=result.model,
                    note="saved from run_cam",
                )
                print(f"saved id={rid} path={path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("bye")


if __name__ == "__main__":
    main()
