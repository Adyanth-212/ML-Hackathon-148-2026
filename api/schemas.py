"""Request/response models for the PathSnap API."""
from __future__ import annotations

from pydantic import BaseModel


class TripInfo(BaseModel):
    trip_id: str
    n_observations: int


class PredictRequest(BaseModel):
    trip_id: str
