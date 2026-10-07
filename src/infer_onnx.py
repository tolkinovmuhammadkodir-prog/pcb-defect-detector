"""
infer_onnx.py — Run the exported PCB detector with onnxruntime only (no PyTorch).

Pipeline (everything Ultralytics normally does for you, written out):
  1. letterbox  : resize keeping aspect ratio, pad to a 640x640 square (grey 114)
  2. normalize  : BGR->RGB, /255, HWC->CHW, add batch dim
  3. run        : onnxruntime -> raw output (1, 4 + num_classes, 8400)
  4. decode     : per candidate, class = argmax of scores, confidence = max score
  5. filter     : drop candidates below the confidence threshold
  6. un-letterbox: map boxes from the padded 640x640 frame back to the original image
  7. NMS        : per-class non-maximum suppression to remove duplicate boxes

Confidence default is 0.25, the value chosen in the README's threshold sweep
(best recall, because a missed defect costs more than a false alarm).

CLI:
    python src/infer_onnx.py --model models/best.onnx --image board.jpg --out out.jpg
"""
import argparse
import ast
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_CLASS_NAMES = ["Missing_hole", "Mouse_bite", "Open_circuit",
                       "Short", "Spur", "Spurious_copper"]


# ---------------------------------------------------------------- preprocessing
def letterbox(img: np.ndarray, size: int = 640, pad_value: int = 114
              ) -> Tuple[np.ndarray, float, int, int]:
    """Resize with preserved aspect ratio and pad to size x size.
    Returns (padded image, scale ratio, left pad, top pad)."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, ratio, left, top


def to_tensor(padded_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))      # HWC -> CHW
    return np.ascontiguousarray(tensor[None])      # add batch dim


# --------------------------------------------------------------- postprocessing
def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Greedy NMS: keep the best box, drop boxes overlapping it too much, repeat."""
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_one_to_many(boxes[best], boxes[rest])
        order = rest[overlaps <= iou_thres]
    return keep


def postprocess(output: np.ndarray, ratio: float, left: int, top: int,
                orig_w: int, orig_h: int, class_names: List[str],
                conf_thres: float = 0.25, iou_thres: float = 0.7) -> List[Dict]:
    """Turn raw model output (1, 4+nc, N) into a list of detections in
    original-image pixel coordinates."""
    num_classes = len(class_names)
    if output.ndim != 3 or output.shape[1] != 4 + num_classes:
        raise ValueError(
            f"Unexpected output shape {output.shape}; expected (1, {4 + num_classes}, N). "
            "Re-export with NMS disabled (see src/export_onnx.py)."
        )
    preds = output[0].T                       # (N, 4 + nc)
    scores_all = preds[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    confs = scores_all.max(axis=1)

    mask = confs >= conf_thres
    preds, class_ids, confs = preds[mask], class_ids[mask], confs[mask]
    if preds.shape[0] == 0:
        return []

    # centre-x, centre-y, w, h  ->  x1, y1, x2, y2 (still in 640x640 padded frame)
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # undo letterbox: remove padding, divide by scale, clip to the image
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

    detections: List[Dict] = []
    for cls in np.unique(class_ids):          # class-aware NMS
        idx = np.where(class_ids == cls)[0]
        for k in nms(boxes[idx], confs[idx], iou_thres):
            i = idx[k]
            detections.append({
                "class_id": int(cls),
                "label": class_names[int(cls)],
                "confidence": round(float(confs[i]), 4),
                "box": [round(float(v), 1) for v in boxes[i]],   # x1, y1, x2, y2
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ------------------------------------------------------------------- detector
def _names_from_metadata(session) -> Optional[List[str]]:
    """Ultralytics stores class names in the ONNX metadata as "{0: 'a', 1: 'b'}"."""
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
        return [parsed[i] for i in sorted(parsed)]
    except (ValueError, SyntaxError, KeyError):
        return None


class OnnxDetector:
    def __init__(self, model_path: str, imgsz: int = 640):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # static exports report ints like [1, 3, 640, 640]; use that if present
        self.imgsz = inp.shape[2] if isinstance(inp.shape[2], int) else imgsz
        self.class_names = _names_from_metadata(self.session) or DEFAULT_CLASS_NAMES

    def detect(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> List[Dict]:
        h, w = image_bgr.shape[:2]
        padded, ratio, left, top = letterbox(image_bgr, self.imgsz)
        output = self.session.run(None, {self.input_name: to_tensor(padded)})[0]
        return postprocess(output, ratio, left, top, w, h, self.class_names, conf, iou)


def draw(image_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f'{d["label"]} {d["confidence"]:.2f}', (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="out.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    detector = OnnxDetector(args.model)
    results = detector.detect(image, conf=args.conf)
    for r in results:
        print(r)
    cv2.imwrite(args.out, draw(image, results))
    print(f"{len(results)} detections -> {args.out}")
