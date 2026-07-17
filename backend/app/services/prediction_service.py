import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.prediction import Prediction
from app.services.model_loader import get_model, LABEL_MAP
from app.services.xai_service import GradCAM, GradCAMPP, overlay_heatmap
from app.utils.image_utils import preprocess, to_tensor


def run_prediction(image_bytes: bytes, filename: str, db: Session) -> Prediction:
    model, device = get_model()

    img_np = preprocess(image_bytes, settings.IMG_SIZE)
    tensor = to_tensor(img_np, device)

    target_layer = model.backbone.blocks[-1][-1].conv_pwl
    gradcam   = GradCAM(model, target_layer)
    gradcampp = GradCAMPP(model, target_layer)

    cam1, pred_class, probs = gradcam.generate(tensor.clone())
    cam2, _, _              = gradcampp.generate(tensor.clone())

    stem = Path(filename).stem
    uid  = uuid.uuid4().hex[:8]

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    gradcam_path   = settings.UPLOAD_DIR / f"{stem}_{uid}_gradcam.png"
    gradcampp_path = settings.UPLOAD_DIR / f"{stem}_{uid}_gradcampp.png"

    overlay1 = overlay_heatmap(img_np, cam1)
    overlay2 = overlay_heatmap(img_np, cam2)

    cv2.imwrite(str(gradcam_path),   cv2.cvtColor(overlay1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gradcampp_path), cv2.cvtColor(overlay2, cv2.COLOR_RGB2BGR))

    record = Prediction(
        image_name      = filename,
        predicted_grade = int(pred_class),
        grade_label     = LABEL_MAP[int(pred_class)],
        confidence      = float(probs[pred_class]),
        probabilities   = json.dumps([round(float(p), 4) for p in probs]),
        gradcam_path    = str(gradcam_path),
        gradcampp_path  = str(gradcampp_path),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_all_predictions(db: Session, skip: int = 0, limit: int = 50):
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_prediction_by_id(prediction_id: int, db: Session):
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()
