# === Important libraries ========================
import os
import shutil
import random
from pathlib import Path

# === CONFIG =====================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMG_DIR = PROJECT_ROOT / "data" / "line2yolo" / "images"
SOURCE_ANN_DIR = PROJECT_ROOT / "data" / "line2yolo" / "annotations_voc"
DEST_ROOT = PROJECT_ROOT / "data" / "line2voc"

DEST_IMG_DIR = DEST_ROOT / "JPEGImages"
DEST_ANN_DIR = DEST_ROOT / "Annotations"
DEST_SPLIT_DIR = DEST_ROOT / "ImageSets" / "Main"

SPLIT_RATIO = {"train": 0.7, "val": 0.2, "test": 0.1}
VALID_EXTS = [".jpg", ".jpeg", ".png"]

# === SETUP =======================================
DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
DEST_ANN_DIR.mkdir(parents=True, exist_ok=True)
DEST_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# === GATHER ALL IMAGES WITH MATCHING XMLs ========
all_images = []
for ext in VALID_EXTS:
    all_images.extend(SOURCE_IMG_DIR.glob(f"*{ext}"))

paired_images = []
for img_path in all_images:
    xml_path = SOURCE_ANN_DIR / f"{img_path.stem}.xml"
    if xml_path.exists():
        paired_images.append((img_path, xml_path))

print(f"Found {len(paired_images)} usable image+annotation pairs.")

# === SHUFFLE & SPLIT =========================
random.shuffle(paired_images)
n = len(paired_images)
train_cut = int(n * SPLIT_RATIO["train"])
val_cut = int(n * (SPLIT_RATIO["train"] + SPLIT_RATIO["val"]))

splits = {
    "train": paired_images[:train_cut],
    "val": paired_images[train_cut:val_cut],
    "test": paired_images[val_cut:]
}

# === COPY FILES & WRITE SPLIT FILES ============
for split_name, pairs in splits.items():
    split_file = DEST_SPLIT_DIR / f"{split_name}.txt"
    with open(split_file, "w") as f:
        for img_path, xml_path in pairs:
            shutil.copy(img_path, DEST_IMG_DIR / img_path.name)
            shutil.copy(xml_path, DEST_ANN_DIR / xml_path.name)
            f.write(f"{img_path.stem}\n")

print("\n Pascal VOC structure created in 'data/line2voc'")
for split_name in splits:
    print(f" {split_name}.txt -> {len(splits[split_name])} files")

