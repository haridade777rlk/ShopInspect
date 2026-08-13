"""Start ShopInspect API + dashboard.
  python scripts/run_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from app.config import load_settings


def main() -> None:
    settings = load_settings()
    print(
        f"[{settings.app_name}] http://{settings.host}:{settings.port}/  "
        f"docs=/docs"
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()
