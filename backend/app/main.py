from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.model import load_model_and_classes

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"], ml["class_names"] = load_model_and_classes()
    print("Model loaded and ready")
    yield
    ml.clear()

app = FastAPI(
    title="Traffic Sign Classifier API",
    description="CNN-based traffic sign classification with Grad-CAM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.predict import router
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml}
