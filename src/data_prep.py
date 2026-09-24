"""
data_prep.py — Downloads the PCB defect dataset and converts it from
Pascal VOC annotation format (xmin/ymin/xmax/ymax, raw pixels) to YOLO
format (center_x/center_y/width/height, normalized 0-1), then splits
into train/val sets in YOLO's expected folder structure.

Verified: 693/693 images converted successfully, 0 skipped.
"""
import os
import random
import shutil
import logging
import xml.etree.ElementTree as ET
import kagglehub

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S", force=True)
logger = logging.getLogger(__name__)

CLASS_TO_ID = {
    "missing_hole": 0, "mouse_bite": 1, "open_circuit": 2,
    "short": 3, "spur": 4, "spurious_copper": 5,
}
CLASS_NAMES_YAML = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']


def voc_to_yolo(xml_path: str, class_to_id: dict) -> list:
    """
    Converts one Pascal VOC XML annotation into a list of YOLO-format
    label lines. The coordinate math (subtract for width/height, add
    half-width/height for center, divide by image size to normalize)
    was hand-derived and verified against manual calculation before
    being trusted on the full dataset.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip().lower()
        class_id = class_to_id[class_name]

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        box_width = xmax - xmin
        box_height = ymax - ymin
        center_x = xmin + (box_width / 2)
        center_y = ymin + (box_height / 2)

        center_x_norm = center_x / img_width
        center_y_norm = center_y / img_height
        width_norm = box_width / img_width
        height_norm = box_height / img_height

        yolo_lines.append(f"{class_id} {center_x_norm:.4f} {center_y_norm:.4f} {width_norm:.4f} {height_norm:.4f}")

    return yolo_lines


def download_and_convert(output_dir: str = "/content/pcb_yolo_dataset") -> dict:
    """Downloads the PCB dataset and converts every annotation to YOLO format."""
    path = kagglehub.dataset_download("akhatova/pcb-defects")
    logger.info(f"Dataset path: {path}")

    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    classes = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    converted_count = 0
    skipped_count = 0

    for class_folder in classes:
        ann_dir = os.path.join(path, "PCB_DATASET", "Annotations", class_folder)
        img_dir = os.path.join(path, "PCB_DATASET", "images", class_folder)

        for xml_file in os.listdir(ann_dir):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(ann_dir, xml_file)
            stem = os.path.splitext(xml_file)[0]
            img_path = os.path.join(img_dir, stem + ".jpg")

            if not os.path.exists(img_path):
                logger.warning(f"No matching image for {xml_file}, skipping")
                skipped_count += 1
                continue

            try:
                yolo_lines = voc_to_yolo(xml_path, CLASS_TO_ID)
            except Exception as e:
                logger.warning(f"Failed to convert {xml_file}: {e}")
                skipped_count += 1
                continue

            with open(os.path.join(labels_out, stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            shutil.copy(img_path, os.path.join(images_out, stem + ".jpg"))
            converted_count += 1

    logger.info(f"Conversion complete: {converted_count} converted, {skipped_count} skipped")
    return {"converted": converted_count, "skipped": skipped_count}


def split_train_val(base: str = "/content/pcb_yolo_dataset", val_split: float = 0.2, seed: int = 42):
    """Splits the converted dataset into train/val, in YOLO's expected folder layout."""
    random.seed(seed)
    all_images = [f for f in os.listdir(os.path.join(base, "images")) if f.endswith(".jpg")]
    random.shuffle(all_images)

    split_idx = int((1 - val_split) * len(all_images))
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    for split_name, files in [("train", train_files), ("val", val_files)]:
        os.makedirs(os.path.join(base, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(base, "labels", split_name), exist_ok=True)
        for img_file in files:
            stem = os.path.splitext(img_file)[0]
            shutil.move(os.path.join(base, "images", img_file), os.path.join(base, "images", split_name, img_file))
            shutil.move(os.path.join(base, "labels", stem + ".txt"), os.path.join(base, "labels", split_name, stem + ".txt"))

    logger.info(f"Train: {len(train_files)} | Val: {len(val_files)}")
    return {"train": len(train_files), "val": len(val_files)}


def write_data_yaml(base: str = "/content/pcb_yolo_dataset") -> str:
    """Writes the YOLO training config pointing at the split dataset."""
    content = f"""train: {base}/images/train
val: {base}/images/val
nc: 6
names: {CLASS_NAMES_YAML}
"""
    yaml_path = f"{base}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(content)
    logger.info(content)
    return yaml_path


if __name__ == "__main__":
    download_and_convert()
    split_train_val()
    write_data_yaml()
