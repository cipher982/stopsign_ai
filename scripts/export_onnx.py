"""One-time script to export a YOLO .pt model to ONNX format.

Usage:
    uv run python scripts/export_onnx.py [model_name]

Default exports yolo11x.pt → yolo11x.onnx in the current directory.
Upload the result to a GitHub release for Dockerfile consumption:
    gh release create v0.2.0-onnx yolo11x.onnx --title "YOLO11x ONNX model"
"""

import sys

from ultralytics import YOLO


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo11x.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    print("Exporting to ONNX (opset 17, imgsz 640, FP32)...")
    model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, half=False)
    print("Done. Output: yolo11x.onnx")


if __name__ == "__main__":
    main()
