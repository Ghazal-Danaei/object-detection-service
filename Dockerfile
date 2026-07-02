# Self-contained CPU image. For GPU inference, switch the base image to an
# NVIDIA CUDA runtime (e.g. nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04),
# install Python, and use onnxruntime-gpu in requirements.txt.
FROM python:3.11-slim

# System libraries required by Pillow / OpenCV (pulled in by ultralytics).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Download the pretrained weights and export to ONNX at build time, so the
# resulting image runs the service without any runtime downloads.
RUN python -m scripts.export_onnx --weights yolov8n.pt --output models

ENV MODEL_BACKEND=onnx \
    ONNX_PATH=models/yolov8n.onnx \
    CONF_THRESHOLD=0.25 \
    IOU_THRESHOLD=0.45 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
