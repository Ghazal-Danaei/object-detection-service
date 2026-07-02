"""Pydantic response models for the detection API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2] in original-image pixels")
    score: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    detections: List[DetectionOut]
    count: int
    image_width: int
    image_height: int
    inference_ms: float
    backend: str
