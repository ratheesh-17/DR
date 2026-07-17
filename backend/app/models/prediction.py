from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from app.db.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, index=True)
    image_name       = Column(String(255), nullable=False)
    predicted_grade  = Column(Integer, nullable=False)
    grade_label      = Column(String(50), nullable=False)
    confidence       = Column(Float, nullable=False)
    probabilities    = Column(Text, nullable=False)   # JSON string of all 5 class probs
    gradcam_path     = Column(String(500), nullable=True)
    gradcampp_path   = Column(String(500), nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
