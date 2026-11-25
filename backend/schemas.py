"""Pydantic schemas describing the public contract of the API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Probabilities(BaseModel):
    NORMAL: float = Field(..., ge=0.0, le=1.0)
    PNEUMONIA: float = Field(..., ge=0.0, le=1.0)


class ModelInfo(BaseModel):
    name: str
    backbone: str
    img_size: List[int]
    class_names: List[str]
    positive_class: str
    threshold: float
    created_at: Optional[str] = None


class ImageInfo(BaseModel):
    filename: Optional[str] = None
    width: int
    height: int
    mode: Optional[str] = None
    format: Optional[str] = None
    size_bytes: int


class PredictionResponse(BaseModel):
    request_id: str
    prediction: str = Field(..., description="'NORMAL' or 'PNEUMONIA'")
    label: str = Field(..., description="Human readable class name")
    is_pneumonia: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the returned label")
    probability: float = Field(..., ge=0.0, le=1.0, description="P(PNEUMONIA)")
    probabilities: Probabilities
    threshold: float
    severity: str = Field(..., description="low | moderate | high - derived from P(PNEUMONIA)")
    inference_ms: float
    processing_ms: float
    image: ImageInfo
    model: ModelInfo
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    detail: Optional[str] = None
    model: Optional[ModelInfo] = None
    tensorflow_version: Optional[str] = None
    api_version: str


class MetricsResponse(BaseModel):
    available: bool
    report: Optional[Dict] = None
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None
