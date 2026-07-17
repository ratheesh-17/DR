from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class PredictionResponse(BaseModel):
    id: int
    image_name: str
    predicted_grade: int
    grade_label: str
    confidence: float
    probabilities: List[float]
    gradcam_url: Optional[str]
    gradcampp_url: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionListItem(BaseModel):
    id: int
    image_name: str
    predicted_grade: int
    grade_label: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True
