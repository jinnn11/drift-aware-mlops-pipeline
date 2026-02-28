from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(min_length=1)


class PredictionItem(BaseModel):
    label: int
    score: float


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]
    model_version: str
    request_id: str
