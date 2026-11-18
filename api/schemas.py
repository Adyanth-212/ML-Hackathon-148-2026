"""Request and response schemas for the PathSnap API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TripInfo(BaseModel):
    trip_id: str
    n_observations: int
    duration_s: float = Field(default=0.0, ge=0.0)


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)
