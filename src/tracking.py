"""
tracking.py — Multi-object tracking using ByteTrack, built on top of a
YOLO detector. Demonstrates the Kalman filter + Hungarian matching
mechanism (via ByteTrack) for maintaining consistent object IDs across
video frames.

Key findings from testing (see README for full writeup):
- ID switches occurred primarily during crossing-paths events, consistent
  with known Hungarian-matching ambiguity when two tracked objects'
  predicted positions become close together.
- This limitation PERSISTED even when upgrading from YOLOv8n to YOLOv8s
  (a more accurate detector) — evidence that the limitation lives in the
  motion-only matching logic itself, not detector weakness. Appearance-
  based re-identification (comparing visual embeddings, not just motion)
  would be the natural next step to address this.
- General COCO-pretrained detection classes are unreliable on
  out-of-distribution objects (e.g. food items on a factory conveyor were
  inconsistently labeled as "cat", "cake", "bowl", "keyboard" frame to
  frame) — reinforcing why domain-specific fine-tuning (as done for the
  PCB defect detector in this same repo) is necessary for real deployment,
  rather than relying on a general pretrained model's classes directly.
"""
from ultralytics import YOLO


def track_video(model_path: str, video_source: str, conf: float = 0.25,
                 tracker: str = "bytetrack.yaml", save: bool = True):
    """
    Runs multi-object tracking on a video using the specified YOLO model
    and tracker.

    Args:
        model_path: path or name of the YOLO weights to use
            (e.g. "yolov8s.pt" for general COCO classes, or a path to a
            custom fine-tuned model like the PCB defect detector).
        video_source: path to the video file, or a supported stream URL.
        conf: confidence threshold below which detections are discarded
            before being passed to the tracker.
        tracker: which tracking algorithm config to use. "bytetrack.yaml"
            uses Kalman filtering + Hungarian matching for motion-based
            tracking; explicitly specifying this (rather than omitting it)
            is important, since Ultralytics silently falls back to a
            different default tracker (BoT-SORT) otherwise.
        save: whether to write an annotated output video to disk.

    Returns:
        The raw Ultralytics results object for further inspection.
    """
    model = YOLO(model_path)
    results = model.track(
        source=video_source,
        conf=conf,
        tracker=tracker,
        save=save,
    )
    return results


if __name__ == "__main__":
    track_video(model_path="yolov8s.pt", video_source="vtest.avi")
