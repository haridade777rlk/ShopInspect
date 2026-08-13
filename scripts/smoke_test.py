"""Non-GUI smoke: synthetic image detect + db + health helpers.
  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.config import load_settings
from app.db import get_record, init_db, insert_record, list_records
from app.detector import Detector


def main() -> None:
    settings = load_settings()
    init_db(settings)
    print("load model...")
    det = Detector(settings)
    det.load()

    # synthetic colorful image (may detect nothing — still validates pipeline)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(img, (450, 240), 80, (0, 0, 255), -1)
    sample = settings.inputs_path / "smoke_sample.jpg"
    cv2.imwrite(str(sample), img)

    result = det.predict_bgr(img, source="file", note="smoke")
    det.save_annotated(result, prefix="smoke")
    rid = insert_record(
        created_at=result.created_at,
        source=result.source,
        image_path=result.image_path,
        num_detections=result.num_detections,
        detections=result.detections,
        model=result.model,
        note="smoke",
    )
    row = get_record(rid)
    rows = list_records(limit=5)
    print("record_id", rid)
    print("num_detections", result.num_detections)
    print("image_path", result.image_path)
    print("db_row_ok", row is not None and row["id"] == rid)
    print("list_n", len(rows))
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
