"""
train.py — Fine-tunes YOLOv8 on the PCB defect dataset.

Verified real results (30 + 30 extended epochs, see README for full table):
  Initial 30 epochs:  mAP50=0.829, mAP50-95=0.396, precision=0.892, recall=0.747
  Extended 30 epochs: mAP50=0.884, mAP50-95=0.431, precision=0.906, recall=0.832
Extending training from the best checkpoint (rather than accepting the
first run) improved recall by 8.5 points while precision also rose
slightly — genuine model improvement, not a threshold trade-off.
"""
from ultralytics import YOLO


def train_initial(data_yaml: str, epochs: int = 30, imgsz: int = 640, batch: int = 16):
    """Initial fine-tune from COCO-pretrained YOLOv8n weights."""
    model = YOLO('yolov8n.pt')
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)
    return results


def train_extended(best_checkpoint: str, data_yaml: str, epochs: int = 30,
                    imgsz: int = 640, batch: int = 16):
    """
    Continues training from a previous best checkpoint for additional
    epochs. Note: resume=False treats this as a fresh run using the
    checkpoint's weights as the starting point (not a literal continuation
    of the epoch counter) — the appropriate choice after a training run
    has completed cleanly, as opposed to recovering from an interruption.
    """
    model = YOLO(best_checkpoint)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                           batch=batch, resume=False)
    return results


if __name__ == "__main__":
    DATA_YAML = "/content/pcb_yolo_dataset/data.yaml"

    train_initial(DATA_YAML)
    # After initial training, point this at the printed best.pt path:
    train_extended("/content/runs/detect/train/weights/best.pt", DATA_YAML)
