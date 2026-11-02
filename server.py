import asyncio
from contextlib import asynccontextmanager, suppress
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import load_model, predict_batch


async def batch_worker(app: FastAPI):
    while True:
        batch = []
        try:
            for _ in range(5):
                batch.append(await app.state.queue.get())
            images, limits, futures = zip(*batch)
            results = await asyncio.to_thread(
                predict_batch, list(images), app.state.loaded, max(limits)
            )
            for result, limit, future in zip(results, limits, futures):
                if not future.done():
                    future.set_result(result[:limit])
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(error)
        finally:
            for _, _, future in batch:
                if not future.done():
                    future.cancel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded = load_model(
        os.getenv("MODEL_DEVICE", "cpu"),
        compile_model=os.getenv("MODEL_COMPILE", "1") == "1",
        quantize=os.getenv("MODEL_QUANTIZE", "1") == "1",
    )
    app.state.queue = asyncio.Queue()
    worker = asyncio.create_task(batch_worker(app))
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        while not app.state.queue.empty():
            _, _, future = app.state.queue.get_nowait()
            future.cancel()


app = FastAPI(title="EfficientNet-B7 Image Classification", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "efficientnet_b7"}


@app.post("/predict")
async def classify(file: UploadFile = File(), top_k: int = Query(5, ge=1, le=1000)):
    if file.file.seek(0, 2) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    file.file.seek(0)
    try:
        with Image.open(file.file) as image:
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or unsupported image")
    future = asyncio.get_running_loop().create_future()
    await app.state.queue.put((rgb, top_k, future))
    predictions = await future
    return {"predictions": predictions}
