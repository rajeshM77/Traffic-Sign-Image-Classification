from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from app.model import preprocess_image, generate_gradcam
import numpy as np

router = APIRouter()

class Prediction(BaseModel):
    label: str
    confidence: float
    class_id: int
    top5: List[dict]
    gradcam_b64: str

@router.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    from app.main import ml
    model      = ml["model"]
    class_names = ml["class_names"]

    image_bytes = await file.read()
    img_array   = preprocess_image(image_bytes)

    preds = model.predict(img_array)[0]               # shape (43,)
    top5_idx = np.argsort(preds)[::-1][:5]

    top5 = [
        {"class_id": int(i), "label": class_names[str(i)], "confidence": round(float(preds[i]) * 100, 2)}
        for i in top5_idx
    ]

    best_idx  = int(top5_idx[0])
    gradcam   = generate_gradcam(model, img_array, best_idx)

    return Prediction(
        label=class_names[str(best_idx)],
        confidence=round(float(preds[best_idx]) * 100, 2),
        class_id=best_idx,
        top5=top5,
        gradcam_b64=gradcam
    )
