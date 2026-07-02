"""FastAPI service exposing the object detector over REST.

Endpoints:
    GET  /health   -> liveness + active backend
    POST /detect   -> multipart image upload, returns detections as JSON

Configuration via environment variables:
    MODEL_BACKEND   "onnx" (default) or "torch"
    ONNX_PATH       path to the .onnx file (default: models/yolov8n.onnx)
    TORCH_WEIGHTS   ultralytics weights name (default: yolov8n.pt)
    CONF_THRESHOLD  confidence threshold (default: 0.25)
    IOU_THRESHOLD   NMS IoU threshold (default: 0.45)
"""
from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .schemas import DetectionOut, DetectionResponse

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx")
ONNX_PATH = os.getenv("ONNX_PATH", "models/yolov8n.onnx")
TORCH_WEIGHTS = os.getenv("TORCH_WEIGHTS", "yolov8n.pt")
CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model exactly once, when the service starts.
    if MODEL_BACKEND == "torch":
        from .detector import TorchDetector
        state["detector"] = TorchDetector(TORCH_WEIGHTS, conf=CONF, iou=IOU)
    else:
        from .onnx_detector import OnnxDetector
        state["detector"] = OnnxDetector(ONNX_PATH, conf=CONF, iou=IOU)
    state["backend"] = MODEL_BACKEND
    yield
    state.clear()


app = FastAPI(title="Object Detection Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "backend": state.get("backend", MODEL_BACKEND)}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detector = state["detector"]
    start = time.perf_counter()
    detections = detector.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return DetectionResponse(
        detections=[DetectionOut(**d.as_dict()) for d in detections],
        count=len(detections),
        image_width=image.width,
        image_height=image.height,
        inference_ms=round(elapsed_ms, 2),
        backend=state["backend"],
    )
