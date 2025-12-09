"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    TRANSFORMER = "transformer"


class SentimentLabel(str, Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


class TextInput(BaseModel):
    """Single text input for prediction."""
    text: str = Field(..., min_length=1, max_length=5000, description="Amharic text to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "ጥሩ ስራ ነው ተባረኩ"
            }
        }


class BatchTextInput(BaseModel):
    """Batch text input for multiple predictions."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "ጥሩ ስራ ነው ተባረኩ",
                    "በጣም መጥፎ ውሳኔ ነው"
                ]
            }
        }


class SentimentResult(BaseModel):
    """Single sentiment prediction result."""
    text: str = Field(..., description="Original input text")
    sentiment: SentimentLabel = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probability: float = Field(..., ge=0, le=1, description="Probability of positive sentiment")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "ጥሩ ስራ ነው ተባረኩ",
                "sentiment": "positive",
                "confidence": 0.92,
                "probability": 0.92
            }
        }


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    success: bool = Field(default=True)
    result: SentimentResult
    model_type: str = Field(..., description="Model used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    success: bool = Field(default=True)
    results: List[SentimentResult]
    model_type: str = Field(..., description="Model used for prediction")
    total_count: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Currently loaded model type")


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
