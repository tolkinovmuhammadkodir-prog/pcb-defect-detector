"""
evaluate_tracking.py — Quantifies tracking quality automatically, rather
than relying on visual inspection.

Measures ID switches: cases where a track's ID changes between
consecutive frames despite spatial continuity (the detection in frame N+1
is very close to where frame N's detection was, but was assigned a
different ID). This turns "I noticed switches while watching the video"
into a real, reproducible, countable metric.
"""
from ultralytics import YOLO
import numpy as np


def compute_iou(box1, box2):
    """Standard IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def count_id_switches(model_path: str, video_source: str, conf: float = 0.25,
                       iou_threshold: float = 0.5):
    """
    Runs tracking and counts likely ID switches: frame-to-frame box pairs
    with high spatial overlap (IoU >= iou_threshold, meaning "almost
    certainly the same physical object") but DIFFERENT track IDs.

    Returns:
        dict with total_frames, total_switches, and switches_per_100_frames
        (a rate, so results are comparable across videos of different length).
    """
    model = YOLO(model_path)
    results = model.track(source=video_source, conf=conf,
                           tracker="bytetrack.yaml", save=False, stream=True)

    prev_boxes = []  # list of (box, track_id) from the previous frame
    switch_count = 0
    frame_count = 0

    for r in results:
        frame_count += 1
        if r.boxes.id is None:
            prev_boxes = []
            continue

        current_boxes = list(zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.id.cpu().numpy().astype(int)))

        for curr_box, curr_id in current_boxes:
            for prev_box, prev_id in prev_boxes:
                iou = compute_iou(curr_box, prev_box)
                if iou >= iou_threshold and curr_id != prev_id:
                    switch_count += 1

        prev_boxes = current_boxes

    rate = (switch_count / frame_count * 100) if frame_count > 0 else 0

    return {
        "total_frames": frame_count,
        "total_switches": switch_count,
        "switches_per_100_frames": round(rate, 2),
    }


if __name__ == "__main__":
    for model_name in ["yolov8n.pt", "yolov8s.pt"]:
        result = count_id_switches(model_name, "vtest.avi")
        print(f"{model_name}: {result}")
