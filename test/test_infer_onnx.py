"""
Unit tests for the ONNX inference code that need no model file and no PyTorch.
Run:  python tests/test_infer_onnx.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from infer_onnx import DEFAULT_CLASS_NAMES, letterbox, nms, postprocess, to_tensor  # noqa: E402

NC = len(DEFAULT_CLASS_NAMES)


def make_output(rows):
    """rows: list of (cx, cy, w, h, class_id, score) in 640x640 padded coords
    -> raw model output of shape (1, 4 + NC, N)."""
    out = np.zeros((1, 4 + NC, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, 0:4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


def test_letterbox_geometry():
    img = np.zeros((300, 600, 3), dtype=np.uint8)           # h=300, w=600
    padded, ratio, left, top = letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert abs(ratio - 640 / 600) < 1e-9
    assert left == 0 and top == (640 - 320) // 2


def test_tensor_shape_and_range():
    t = to_tensor(np.full((640, 640, 3), 255, dtype=np.uint8))
    assert t.shape == (1, 3, 640, 640) and t.dtype == np.float32
    assert abs(float(t.max()) - 1.0) < 1e-6


def test_boxes_map_back_to_original_image():
    # Original 600x300 image -> ratio 640/600, left 0, top 160.
    ratio, left, top = 640 / 600, 0, 160
    # Object whose true box in the ORIGINAL image is x 100-200, y 50-150.
    x1, y1, x2, y2 = 100, 50, 200, 150
    cx = (x1 + x2) / 2 * ratio + left
    cy = (y1 + y2) / 2 * ratio + top
    w, h = (x2 - x1) * ratio, (y2 - y1) * ratio
    dets = postprocess(make_output([(cx, cy, w, h, 3, 0.9)]),
                       ratio, left, top, 600, 300, DEFAULT_CLASS_NAMES)
    assert len(dets) == 1 and dets[0]["label"] == "Short"
    for got, want in zip(dets[0]["box"], (x1, y1, x2, y2)):
        assert abs(got - want) < 0.2, (dets[0]["box"], (x1, y1, x2, y2))


def test_confidence_threshold_filters():
    out = make_output([(100, 100, 40, 40, 0, 0.30), (300, 300, 40, 40, 1, 0.10)])
    dets = postprocess(out, 1.0, 0, 0, 640, 640, DEFAULT_CLASS_NAMES, conf_thres=0.25)
    assert [d["label"] for d in dets] == ["Missing_hole"]


def test_nms_removes_duplicates_same_class_only():
    rows = [
        (100, 100, 50, 50, 2, 0.90),   # keep
        (104, 102, 50, 50, 2, 0.60),   # same class, heavy overlap -> removed
        (104, 102, 50, 50, 4, 0.55),   # different class, same place -> kept
        (400, 400, 50, 50, 2, 0.80),   # far away -> kept
    ]
    dets = postprocess(make_output(rows), 1.0, 0, 0, 640, 640, DEFAULT_CLASS_NAMES)
    assert len(dets) == 3
    assert sorted(d["label"] for d in dets) == ["Open_circuit", "Open_circuit", "Spur"]


def test_empty_and_bad_shape():
    out = make_output([(100, 100, 40, 40, 0, 0.05)])
    assert postprocess(out, 1.0, 0, 0, 640, 640, DEFAULT_CLASS_NAMES) == []
    try:
        postprocess(np.zeros((1, 9, 10), np.float32), 1.0, 0, 0, 640, 640, DEFAULT_CLASS_NAMES)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_nms_function_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [1, 2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
