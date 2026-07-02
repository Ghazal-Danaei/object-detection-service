# Object Detection Service — YOLOv8 → ONNX → FastAPI → Docker

An end-to-end object-detection micro-service that takes a model from PyTorch
weights through ONNX optimization to a containerized REST API. 

[![CI](https://github.com/USERNAME/object-detection-service/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/object-detection-service/actions/workflows/ci.yml)

> Replace `USERNAME` in the badge URL with your GitHub username after pushing.

---



## Architecture

```
            ┌──────────────┐   export    ┌──────────────┐
            │  YOLOv8 (.pt)│ ──────────► │  model.onnx  │
            │  Ultralytics │   ONNX      │              │
            └──────────────┘             └──────┬───────┘
                                                │ onnxruntime
   image (multipart)                            ▼
   ────────────────►  FastAPI  ──►  letterbox → infer → decode → NMS
                      /detect                    │
                                                 ▼
                              JSON: boxes, scores, class names, latency
```



## Project structure

```
object-detection-service/
├── src/
│   ├── metrics.py         # IoU, NMS, Average Precision (pure NumPy, unit-tested)
│   ├── labels.py          # COCO class names
│   ├── onnx_detector.py   # ONNX Runtime inference + manual pre/post-processing
│   ├── detector.py        # PyTorch (Ultralytics) reference + ONNX export
│   ├── schemas.py         # Pydantic response models
│   └── api.py             # FastAPI app (/health, /detect)
├── scripts/
│   ├── export_onnx.py     # export YOLOv8 weights to ONNX
│   └── benchmark.py       # PyTorch vs ONNX latency + parity check
├── tests/
│   ├── test_metrics.py    # IoU / NMS / AP correctness
│   └── test_api.py        # API request/response (faked detector, no heavy deps)
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Quickstart (local)

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. export the model to ONNX (downloads yolov8n.pt on first run)
python -m scripts.export_onnx --weights yolov8n.pt --output models

# 3. run the API
uvicorn src.api:app --reload
# -> open http://127.0.0.1:8000/docs for the interactive Swagger UI
```

### Call the API

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
     -F "file=@samples/street.jpg"
```

```json
{
  "detections": [
    {"box": [12.4, 88.1, 220.6, 410.0], "score": 0.91, "class_id": 0, "class_name": "person"},
    {"box": [305.0, 150.2, 470.8, 300.4], "score": 0.77, "class_id": 2, "class_name": "car"}
  ],
  "count": 2,
  "image_width": 1280,
  "image_height": 720,
  "inference_ms": 41.3,
  "backend": "onnx"
}
```

### Benchmark PyTorch vs ONNX

```bash
python -m scripts.benchmark --image samples/street.jpg --onnx models/yolov8n.onnx
```

```
====================================================
PyTorch (Ultralytics):  78.42 ms  (+/-  3.10)  | 5 detections
ONNX Runtime:           41.27 ms  (+/-  1.84)  | 5 detections
Speedup (torch/onnx):    1.90x
Output parity (same class, IoU>=0.5): 100.0%
====================================================
```
*(Numbers are illustrative; they depend on your hardware.)*

## Run with Docker

```bash
docker build -t object-detection-service .
docker run -p 8000:8000 object-detection-service
# POST images to http://localhost:8000/detect
```

The image is self-contained — weights are downloaded and the ONNX model is
exported during the build, so the running container needs no external downloads.

## Testing

```bash
pytest tests/ -v
```

The test suite covers the detection metrics (IoU, NMS, AP) and the API contract.
The API tests inject a fake detector, so they run in CI without torch,
onnxruntime, or model weights.



## License

Source code: MIT (see `LICENSE`). The pretrained YOLOv8 weights and the
`ultralytics` package are AGPL-3.0 — review their terms before any commercial use.
