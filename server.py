from contextlib import asynccontextmanager
import os
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import load_model, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded = load_model(os.getenv("MODEL_DEVICE", "cpu"))
    app.state.lock = Lock()
    yield
    del app.state.loaded


app = FastAPI(title="EfficientNet-B7 Image Classification", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "efficientnet_b7"}


@app.post("/predict")
def classify(file: UploadFile = File(), top_k: int = Query(5, ge=1, le=1000)):
    if file.file.seek(0, 2) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    file.file.seek(0)
    try:
        with Image.open(file.file) as image:
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or unsupported image")
    with app.state.lock:
        predictions = predict(rgb, app.state.loaded, top_k)
    return {"predictions": predictions}
