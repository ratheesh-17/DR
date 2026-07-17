import json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.prediction import PredictionResponse, PredictionListItem
from app.services.prediction_service import (
    run_prediction,
    get_all_predictions,
    get_prediction_by_id,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    record = run_prediction(image_bytes, file.filename, db)
    return _to_response(record)


@router.get("/history", response_model=list[PredictionListItem])
def history(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return get_all_predictions(db, skip, limit)


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_one(prediction_id: int, db: Session = Depends(get_db)):
    record = get_prediction_by_id(prediction_id, db)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _to_response(record)


def _to_response(record) -> PredictionResponse:
    base = "/uploads"
    from pathlib import Path
    return PredictionResponse(
        id              = record.id,
        image_name      = record.image_name,
        predicted_grade = record.predicted_grade,
        grade_label     = record.grade_label,
        confidence      = record.confidence,
        probabilities   = json.loads(record.probabilities),
        gradcam_url     = f"{base}/{Path(record.gradcam_path).name}"   if record.gradcam_path   else None,
        gradcampp_url   = f"{base}/{Path(record.gradcampp_path).name}" if record.gradcampp_path else None,
        created_at      = record.created_at,
    )
