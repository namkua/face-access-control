"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction_id: str
    predicted_user: Optional[str]
    confidence: Optional[float]
    face_detected: bool
    processing_time_ms: float
    status: str
    message: Optional[str] = None
    embedding: Optional[List[float]] = None





class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    dependencies: dict


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
