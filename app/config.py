"""Load ShopInspect config from YAML + project root resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    app_name: str = "车间质检台"
    app_name_en: str = "ShopInspect"
    camera_index: int = 0
    camera_backend: str = "dshow"
    model_path: str = "yolo11n.pt"
    confidence: float = 0.4
    iou: float = 0.45
    device: str = "cpu"
    imgsz: int = 640
    max_infer_side: int = 960
    jpeg_quality: int = 85
    db_path: str = "data/shopinspect.db"
    outputs_dir: str = "data/outputs"
    inputs_dir: str = "data/inputs"
    models_dir: str = "models"
    host: str = "127.0.0.1"
    port: int = 8787
    return_annotated_default: bool = True
    project_root: Path = PROJECT_ROOT

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    @property
    def db_file(self) -> Path:
        return self.resolve(self.db_path)

    @property
    def outputs_path(self) -> Path:
        return self.resolve(self.outputs_dir)

    @property
    def inputs_path(self) -> Path:
        return self.resolve(self.inputs_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    def resolved_model_path(self) -> str:
        raw = Path(self.model_path)
        if raw.is_absolute() and raw.exists():
            return str(raw)
        local = self.models_path / raw.name
        if local.exists():
            return str(local)
        candidate = self.resolve(self.model_path)
        if candidate.exists():
            return str(candidate)
        return self.model_path


_settings: Settings | None = None


def load_settings(config_path: str | Path | None = None) -> Settings:
    global _settings
    env_cfg = os.environ.get("SHOPINSPECT_CONFIG")
    path = Path(config_path or env_cfg or PROJECT_ROOT / "config.yaml")
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    known = {f.name for f in fields(Settings)} - {"project_root"}
    filtered = {k: v for k, v in data.items() if k in known}
    settings = Settings(**filtered)
    settings.project_root = PROJECT_ROOT
    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    settings.inputs_path.mkdir(parents=True, exist_ok=True)
    settings.models_path.mkdir(parents=True, exist_ok=True)
    settings.db_file.parent.mkdir(parents=True, exist_ok=True)
    _settings = settings
    return settings


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        return load_settings()
    return _settings
