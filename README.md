# PCB Defect Detection + Multi-Object Tracking

A YOLOv8-based object detector for identifying manufacturing defects on
printed circuit boards, extended with a multi-object tracking analysis
demonstrating motion-based tracking (Kalman filtering + Hungarian
matching via ByteTrack) and its real limitations.

**[Live Demo](#)** ← add after deployment

## Problem

PCB manufacturing requires catching defects — missing drill holes, short
circuits, broken copper traces — before boards ship. This is a genuine
industrial computer vision problem (manufacturing defect inspection is a
commonly requested skill in current CV job postings), with 6 distinct
defect types and multiple defects often present per board image.

## Part 1: Defect Detection

### Approach

- **Format conversion**: the dataset's annotations were in Pascal VOC
  format (corner coordinates, raw pixels) — YOLO requires center
  coordinates normalized 0-1. The conversion math was derived by hand on
  a real example first, then automated and verified against the manual
  calculation before running on the full dataset.
- **Transfer learning**: YOLOv8n fine-tuned from COCO-pretrained weights.
- **Extended training**: after an initial 30-epoch run, training was
  continued for 30 more epochs from the best checkpoint — this produced
  a genuine improvement in both precision and recall, not just a
  threshold trade-off (see Results).
- **Deliberate threshold selection**: rather than accepting YOLO's
  default confidence threshold, a sweep across multiple values was run
  and a threshold was chosen based on the cost asymmetry of the use case
  (missing a defect is costlier than a false alarm).

### Results

**Dataset**: 693 images converted (0 skipped) — 554 train / 139 validation.

**Training progression** (verified, both runs actually executed):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| After 30 epochs | 0.892 | 0.747 | 0.829 | 0.396 |
| After 30 more epochs (from best checkpoint) | 0.906 | 0.832 | 0.884 | 0.431 |

Extending training improved recall by **8.5 points** while precision also
rose — genuine model improvement, not a threshold trade-off. The weakest
class (`spur`) saw the largest gain, confirming the model hadn't fully
converged at 30 epochs.

**Final per-class results** (best model, default conf):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Missing_hole | 0.978 | 0.974 | 0.980 | 0.521 |
| Mouse_bite | 0.891 | 0.785 | 0.862 | 0.389 |
| Open_circuit | 0.855 | 0.773 | 0.870 | 0.406 |
| Short | 0.911 | 0.904 | 0.943 | 0.467 |
| Spur | 0.926 | 0.723 | 0.810 | 0.388 |
| Spurious_copper | 0.876 | 0.829 | 0.841 | 0.415 |

**Confidence threshold sweep** (on the final model):

| conf | Precision | Recall |
|---|---|---|
| 0.15 | 0.904 | 0.828 |
| **0.25 (chosen)** | 0.886 | **0.841** |
| 0.35 | 0.927 | 0.800 |
| 0.50 | 0.965 | 0.758 |

**`conf=0.25` was selected deliberately**: it achieves the highest recall
of any tested threshold while keeping precision strong — the right choice
given that in manufacturing QC, missing a real defect (false negative) is
costlier than a false alarm (false positive) that just triggers a manual
recheck.

## Part 2: Multi-Object Tracking

### Approach

ByteTrack (Kalman filtering for motion prediction + Hungarian matching
for optimal frame-to-frame assignment) was tested on two video sources to
evaluate tracking behavior, using both YOLOv8n and YOLOv8s as the
underlying detector.

### Findings

- **ID switches occurred primarily during crossing-paths events** —
  consistent with the known limitation of motion-only matching: when two
  tracked objects' predicted positions become close together, the
  Hungarian assignment can become ambiguous.
- **This limitation persisted even with a more accurate detector
  (YOLOv8s vs YOLOv8n)**, indicating the limitation lives in the
  motion-only matching logic itself, not detector weakness — a better
  detector alone cannot fully resolve it.
- **General COCO-pretrained classes are unreliable on out-of-distribution
  objects**: tested on real factory conveyor footage of food products,
  the model's label fluctuated between "cat", "cake", "bowl", "keyboard",
  and "sports ball" frame to frame, since COCO has no relevant category
  for the actual objects. This directly reinforces why domain-specific
  fine-tuning (as done for the PCB detector above) is necessary for real
  deployment — a general pretrained model cannot be trusted on classes
  it wasn't built to recognize.
- An automated ID-switch counter (`evaluate_tracking.py`) was built to
  quantify this rather than relying on visual inspection — counting
  frame-to-frame box pairs with high spatial overlap (IoU ≥ 0.5,
  indicating the same physical object) but different track IDs.

### Multi-Camera Re-Identification (Design, Not Implemented)

The natural next step — re-identifying the same object across two
different camera feeds, which pure motion-based tracking cannot do since
there's no shared motion history between separate cameras — was scoped
out to protect project timeline, but the approach is:

1. Extract a visual embedding per detected object using a pretrained CNN
   backbone (e.g. the same ResNet-18 architecture used in the skin lesion
   classifier project, taking the penultimate layer's output).
2. Compare embeddings across camera feeds using cosine similarity.
3. Match candidates above a similarity threshold (e.g. ~0.8) as the same
   physical object, independent of either camera's motion history.

## Project Structure

```
pcb-defect-detector/
├── src/
│   ├── data_prep.py        # VOC->YOLO conversion, train/val split, data.yaml
│   ├── train.py             # Initial + extended training
│   ├── evaluate.py          # Confidence threshold sweep
│   ├── tracking.py          # ByteTrack multi-object tracking
│   └── evaluate_tracking.py # Automated ID-switch counting
├── requirements.txt
└── README.md
```

## Running It

```bash
pip install -r requirements.txt

python src/data_prep.py   # download + convert + split
python src/train.py       # initial + extended training
python src/evaluate.py    # confidence threshold sweep
python src/tracking.py    # tracking demo
python src/evaluate_tracking.py  # quantified ID-switch analysis
```

## Tech Stack

YOLOv8 (Ultralytics) · ByteTrack · OpenCV · PyTorch

## Limitations & Future Work

- Multi-camera re-identification (design above) not yet implemented.
- `spur` and `open_circuit` remain the weakest classes — targeted data
  augmentation or additional training epochs weighted toward these
  classes is a concrete next step.
- Not validated for real production QC use — a research/portfolio
  project, not a certified inspection system.
