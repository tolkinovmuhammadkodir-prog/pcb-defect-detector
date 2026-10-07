"""
check_onnx_parity.py — Does the ONNX model give the same detections as the
original PyTorch model? Run this once after exporting (needs ultralytics).

    python tests/check_onnx_parity.py --weights best.pt --onnx models/best.onnx \
        --images /content/pcb_yolo_dataset/images/val --limit 30

For every image it runs both models at the same confidence and counts how many
PyTorch detections have an ONNX detection of the same class with IoU >= 0.9.
Small differences are expected (ONNX path pads to a fixed 640x640 square; the
Ultralytics path may pad less), so judge the match rate, not exact equality.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from infer_onnx import OnnxDetector, iou_one_to_many  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", default="models/best.onnx")
    parser.add_argument("--images", required=True, help="folder of .jpg/.png images")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    from ultralytics import YOLO
    pt_model = YOLO(args.weights)
    onnx_model = OnnxDetector(args.onnx)

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))[:args.limit]
    pt_total = matched = count_equal = 0
    for p in paths:
        img = cv2.imread(p)
        res = pt_model.predict(img, conf=args.conf, imgsz=640, verbose=False)[0]
        pt_boxes = res.boxes.xyxy.cpu().numpy()
        pt_cls = res.boxes.cls.cpu().numpy().astype(int)
        onnx_dets = onnx_model.detect(img, conf=args.conf)

        pt_total += len(pt_boxes)
        count_equal += int(len(pt_boxes) == len(onnx_dets))
        for box, cls in zip(pt_boxes, pt_cls):
            same = [d for d in onnx_dets if d["class_id"] == cls]
            if same:
                ious = iou_one_to_many(box, np.array([d["box"] for d in same]))
                matched += int(ious.max() >= 0.9)

    print(f"images checked           : {len(paths)}")
    print(f"same detection count     : {count_equal}/{len(paths)} images")
    print(f"PyTorch boxes matched    : {matched}/{pt_total} "
          f"({100 * matched / max(pt_total, 1):.1f}%) with IoU>=0.9 and same class")


if __name__ == "__main__":
    main()
