"""Probe local cameras. Run from project root:
  python scripts/probe_camera.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.camera import probe_cameras
from app.config import load_settings


def main() -> None:
    settings = load_settings()
    print(f"config camera_index={settings.camera_index} backend={settings.camera_backend}")
    rows = probe_cameras(max_index=5, backend=settings.camera_backend)
    for r in rows:
        print(
            f"cam{r['index']}: opened={r['opened']} read_ok={r['read_ok']} shape={r['shape']}"
        )
    ok = any(r["opened"] and r["read_ok"] for r in rows)
    if not ok:
        print("NO usable camera. Check privacy settings / close other apps.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
