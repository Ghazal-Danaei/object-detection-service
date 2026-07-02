"""ONNX Runtime inference for a YOLOv8 detector.

This module re-implements pre/post-processing by hand instead of relying on the
Ultralytics runtime, to demonstrate understanding of what the exported graph
expects and produces:

  * letterbox preprocessing (aspect-preserving resize + padding),
  * decoding the raw YOLOv8 output tensor of shape (1, 4 + num_classes, anchors),
  * mapping boxes back to original-image coordinates, and
  * class-aware non-maximum suppression.

The parity check in scripts/benchmark.py validates this path against the
PyTorch (Ultralytics) path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

try:  # keep the module importable for metric-only test runs
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from .labels import COCO_CLASSES
from .metrics import nms


@dataclass
class Detection:
    """A single detected object in original-image pixel coordinates."""
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    class_id: int
    class_name: str

    def as_dict(self) -> dict:
        return {
            "box": [round(float(v), 2) for v in self.box],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


def letterbox(image: np.ndarray, new_shape: int = 640, color: int = 114):
    """Resize keeping aspect ratio and pad to a square (new_shape x new_shape).

    Returns the padded image plus the (ratio, pad_left, pad_top) needed to map
    predicted boxes back to the original image.
    """
    h, w = image.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    )

    pad_w = new_shape - new_w
    pad_h = new_shape - new_h
    top = pad_h // 2
    left = pad_w // 2

    padded = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = resized
    return padded, ratio, left, top


class OnnxDetector:
    """Run a YOLOv8 ONNX model with NumPy-only post-processing."""

    def __init__(self, onnx_path: str, conf: float = 0.25, iou: float = 0.45,
                 input_size: int = 640):
        if ort is None:
            raise ImportError("onnxruntime is required to run OnnxDetector")
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou
        self.input_size = input_size

    def _preprocess(self, image: Image.Image):
        img = np.asarray(image.convert("RGB"))
        padded, ratio, pad_left, pad_top = letterbox(img, self.input_size)
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # HWC -> NCHW
        return np.ascontiguousarray(blob), ratio, pad_left, pad_top

    def _postprocess(self, output: np.ndarray, ratio, pad_left, pad_top,
                     orig_w, orig_h) -> List[Detection]:
        # YOLOv8 output: (1, 4 + num_classes, anchors) -> (anchors, 4 + num_classes)
        preds = np.squeeze(output, axis=0).T
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confidences >= self.conf
        boxes_cxcywh = boxes_cxcywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        if len(boxes_cxcywh) == 0:
            return []

        # center-x, center-y, w, h (in letterboxed space) -> x1, y1, x2, y2
        cx, cy, bw, bh = boxes_cxcywh.T
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # undo letterbox: subtract padding, then rescale to original image
        x1 = ((x1 - pad_left) / ratio).clip(0, orig_w)
        x2 = ((x2 - pad_left) / ratio).clip(0, orig_w)
        y1 = ((y1 - pad_top) / ratio).clip(0, orig_h)
        y2 = ((y2 - pad_top) / ratio).clip(0, orig_h)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        detections: List[Detection] = []
        for cls in np.unique(class_ids):
            mask = class_ids == cls
            cls_boxes = boxes_xyxy[mask]
            cls_scores = confidences[mask]
            for k in nms(cls_boxes, cls_scores, self.iou):
                cid = int(cls)
                detections.append(
                    Detection(
                        box=tuple(cls_boxes[k].tolist()),
                        score=float(cls_scores[k]),
                        class_id=cid,
                        class_name=COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid),
                    )
                )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def predict(self, image: Image.Image) -> List[Detection]:
        blob, ratio, pad_left, pad_top = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        orig_w, orig_h = image.size
        return self._postprocess(outputs[0], ratio, pad_left, pad_top, orig_w, orig_h)
