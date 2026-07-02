"""Export the YOLOv8 PyTorch model to ONNX.

Usage:
    python -m scripts.export_onnx --weights yolov8n.pt --output models
"""
import argparse

from src.detector import TorchDetector


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--weights", default="yolov8n.pt", help="ultralytics weights")
    parser.add_argument("--output", default="models", help="output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="input size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    detector = TorchDetector(args.weights)
    path = detector.export_onnx(args.output, imgsz=args.imgsz, opset=args.opset)
    print(f"Exported ONNX model to: {path}")


if __name__ == "__main__":
    main()
