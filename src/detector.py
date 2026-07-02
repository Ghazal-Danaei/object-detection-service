"""PyTorch detector wrapper (Ultralytics YOLOv8) and ONNX export helper.

Used as the reference implementation for the benchmark/parity check and as an
optional serving backend (MODEL_BACKEND=torch).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from PIL import Image

from .onnx_detector import Detection  # reuse the same result type

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class TorchDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.45):
        if YOLO is None:
            raise ImportError("ultralytics is required to run TorchDetector")
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou

    def predict(self, image: Image.Image) -> List[Detection]:
        results = self.model.predict(image, conf=self.conf, iou=self.iou, verbose=False)
        result = results[0]
        names = result.names
        detections: List[Detection] = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    box=tuple(xyxy),
                    score=float(box.conf[0]),
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                )
            )
        return detections

    def export_onnx(self, output_dir: str = "models", imgsz: int = 640,
                    opset: int = 12) -> str:
        """Export to ONNX and place the file in ``output_dir``."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        exported = Path(self.model.export(format="onnx", imgsz=imgsz,
                                          opset=opset, dynamic=False))
        target = Path(output_dir) / exported.name
        if exported.resolve() != target.resolve():
            shutil.move(str(exported), str(target))
        return str(target)
