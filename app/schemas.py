"""Pydantic shapes for API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    label: str
    confidence: float
    bbox_xyxy: list[float] = Field(description="[x1,y1,x2,y2]")


class DetectResponse(BaseModel):
    id: Optional[int] = None
    created_at: str
    source: Literal["camera", "upload", "path", "file"]
    image_path: Optional[str] = None
    num_detections: int
    detections: list[DetectionItem]
    model: str
    note: Optional[str] = None
    annotated_base64: Optional[str] = None
    elapsed_ms: float = 0.0
    conf_used: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    labels: Optional[dict[str, int]] = None
    top_label: Optional[str] = None
    avg_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    status: Optional[str] = None
    work_order: Optional[str] = None
    batch_id: Optional[str] = None


class RecordSummary(BaseModel):
    id: int
    created_at: str
    source: str
    image_path: Optional[str]
    num_detections: int
    model: str
    note: Optional[str] = None
    elapsed_ms: Optional[float] = None
    conf_used: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    labels: Optional[dict[str, int]] = None
    top_label: Optional[str] = None
    avg_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    status: Optional[str] = None
    work_order: Optional[str] = None
    batch_id: Optional[str] = None


class RecordDetail(RecordSummary):
    detections: list[DetectionItem]
    raw_json: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    app: str
    app_en: str
    model: str
    device: str
    model_loaded: bool
    camera_index: int
    version: str = "0.1.3"


class PathDetectRequest(BaseModel):
    path: str
    recursive: bool = False
    note: Optional[str] = None
    work_order: Optional[str] = None
    batch_id: Optional[str] = None
    save: bool = True


class StatsResponse(BaseModel):
    total_records: int
    total_detections: int
    by_source: dict[str, int]
    by_label: dict[str, int] = Field(default_factory=dict)
    avg_elapsed_ms: Optional[float] = None
    alert_records: int = 0
    model: str
    device: str
