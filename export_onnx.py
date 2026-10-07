"""
export_onnx.py — Export the trained YOLOv8 PCB detector to ONNX.

Why ONNX: the trained .pt file needs PyTorch + Ultralytics to run (~GBs of
dependencies). An .onnx file runs on onnxruntime alone (a small CPU-friendly
runtime), which is what makes a lightweight Docker image possible.

Usage (e.g. in Colab, after training):
    pip install ultralytics onnx
    python src/export_onnx.py --weights /content/runs/detect/train2/weights/best.pt

Produces best.onnx next to the weights, and copies it to models/best.onnx.
NMS is NOT baked into the graph: the raw output is (1, 4 + num_classes, 8400)
and src/infer_onnx.py applies confidence filtering + NMS itself.
"""
import argparse
import os
import shutil

from ultralytics import YOLO


def export(weights: str, imgsz: int = 640, opset: int = 12, simplify: bool = False) -> str:
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset,
                             simplify=simplify, dynamic=False)
    return str(onnx_path)


def sanity_check(onnx_path: str) -> None:
    """Load with onnxruntime only (no PyTorch) and print what the service will see."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"input : {inp.name} {inp.shape}")
    print(f"output: {out.name} {out.shape}")
    print("class names metadata:", session.get_modelmeta().custom_metadata_map.get("names"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    path = export(args.weights, args.imgsz, args.opset, args.simplify)
    os.makedirs(args.out_dir, exist_ok=True)
    target = os.path.join(args.out_dir, "best.onnx")
    shutil.copy(path, target)
    print(f"Exported: {path}\nCopied to: {target}")
    sanity_check(target)
