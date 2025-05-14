# scripts/convert_yolo_to_voc.py
# This script converts YOLO-format .txt annotations to Pascal VOC .xml annotations

import os
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image

# ─── CONFIGURATION ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "line2yolo" / "labels"
IMAGES_DIR = PROJECT_ROOT / "data" / "line2yolo" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "line2yolo" / "annotations_voc"

CLASS_NAMES = ["object", "object_1"]  # the provided dataset has only 2 classes [0: default, 1: additional object, can be rock,etc...]
IMG_EXTS = [".png", ".jpg", ".jpeg"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── HELPER: Convert YOLO box to Pascal VOC ──────────────────────
def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x, y, w, h = box
    x1 = int((x - w / 2) * size[0])
    y1 = int((y - h / 2) * size[1])
    x2 = int((x + w / 2) * size[0])
    y2 = int((y + h / 2) * size[1])
    return x1, y1, x2, y2

# ─── HELPER: Create VOC-style XML annotation ─────────────────────
def create_voc_xml(img_path, objects, size):
    annotation = Element('annotation')
    SubElement(annotation, 'folder').text = img_path.parent.name
    SubElement(annotation, 'filename').text = img_path.name

    size_tag = SubElement(annotation, 'size')
    SubElement(size_tag, 'width').text = str(size[0])
    SubElement(size_tag, 'height').text = str(size[1])
    SubElement(size_tag, 'depth').text = "3"

    for obj in objects:
        obj_tag = SubElement(annotation, 'object')
        SubElement(obj_tag, 'name').text = obj['class']
        SubElement(obj_tag, 'pose').text = "Unspecified"
        SubElement(obj_tag, 'truncated').text = "0"
        SubElement(obj_tag, 'difficult').text = "0"

        bbox = SubElement(obj_tag, 'bndbox')
        SubElement(bbox, 'xmin').text = str(obj['xmin'])
        SubElement(bbox, 'ymin').text = str(obj['ymin'])
        SubElement(bbox, 'xmax').text = str(obj['xmax'])
        SubElement(bbox, 'ymax').text = str(obj['ymax'])

    return parseString(tostring(annotation)).toprettyxml(indent="  ")

# ─── MAIN CONVERSION LOOP ─────────────────────────────────────────
converted = 0
skipped_empty = 0
skipped_bad_class = 0

for txt_file in LABELS_DIR.glob("*.txt"):
    # Match image file with the same stem and known extensions
    img_file = None
    for ext in IMG_EXTS:
        candidate = IMAGES_DIR / f"{txt_file.stem}{ext}"
        if candidate.exists():
            img_file = candidate
            break

    if img_file is None:
        print(f"[WARN] No matching image for {txt_file.name}")
        continue

    with Image.open(img_file) as img:
        width, height = img.size

    with open(txt_file, "r") as f:
        lines = f.read().strip().splitlines()

    if not lines:
        skipped_empty += 1
        continue  # skip empty annotation files

    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # skip invalid entries

        class_id = int(parts[0])
        if class_id >= len(CLASS_NAMES):
            print(f"[WARN] Unknown class ID {class_id} in file {txt_file.name}")
            skipped_bad_class += 1
            continue

        bbox = list(map(float, parts[1:]))
        xmin, ymin, xmax, ymax = convert_box((width, height), bbox)
        objects.append({
            "class": CLASS_NAMES[class_id],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    if not objects:
        skipped_empty += 1
        continue  # skip writing XML for images with no valid boxes

    xml_content = create_voc_xml(img_file, objects, (width, height))
    output_file = OUTPUT_DIR / f"{txt_file.stem}.xml"
    with open(output_file, "w") as f:
        f.write(xml_content)
    converted += 1

print(f"\nSuccessfully converted {converted} annotation files to Pascal VOC format.")
print(f"XMLs saved to: {OUTPUT_DIR}")
print(f"Skipped {skipped_empty} files (empty or no valid boxes).")
print(f"Skipped {skipped_bad_class} entries with bad class ID.")
