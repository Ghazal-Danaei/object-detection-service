"""Benchmark PyTorch vs ONNX Runtime: latency and output parity.

Demonstrates the value of exporting to ONNX and gives a reproducible way to
confirm the hand-written ONNX post-processing matches the reference PyTorch
output.

Usage:
    python -m scripts.benchmark --image samples/street.jpg --onnx models/yolov8n.onnx
"""
import argparse
import time

import numpy as np
from PIL import Image

from src.detector import TorchDetector
from src.metrics import iou_matrix
from src.onnx_detector import OnnxDetector


def time_predict(detector, image, runs=30, warmup=5):
    for _ in range(warmup):
        detector.predict(image)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        detector.predict(image)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def match_rate(dets_a, dets_b, iou_thr=0.5):
    """Fraction of `dets_a` matched by a `dets_b` box (same class, IoU >= thr)."""
    if not dets_a:
        return 1.0 if not dets_b else 0.0
    if not dets_b:
        return 0.0
    boxes_a = np.array([d.box for d in dets_a])
    boxes_b = np.array([d.box for d in dets_b])
    ious = iou_matrix(boxes_a, boxes_b)
    matched = 0
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            if da.class_id == db.class_id and ious[i, j] >= iou_thr:
                matched += 1
                break
    return matched / len(dets_a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to a test image")
    parser.add_argument("--onnx", default="models/yolov8n.onnx")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    torch_det = TorchDetector(args.weights)
    onnx_det = OnnxDetector(args.onnx)

    t_mean, t_std = time_predict(torch_det, image, runs=args.runs)
    o_mean, o_std = time_predict(onnx_det, image, runs=args.runs)

    torch_results = torch_det.predict(image)
    onnx_results = onnx_det.predict(image)
    parity = match_rate(torch_results, onnx_results)

    speedup = t_mean / o_mean if o_mean > 0 else float("nan")
    print("=" * 52)
    print(f"PyTorch (Ultralytics): {t_mean:6.2f} ms  (+/- {t_std:5.2f})  "
          f"| {len(torch_results)} detections")
    print(f"ONNX Runtime:          {o_mean:6.2f} ms  (+/- {o_std:5.2f})  "
          f"| {len(onnx_results)} detections")
    print(f"Speedup (torch/onnx):  {speedup:5.2f}x")
    print(f"Output parity (same class, IoU>=0.5): {parity * 100:.1f}%")
    print("=" * 52)


if __name__ == "__main__":
    main()
