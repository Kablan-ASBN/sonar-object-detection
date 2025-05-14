# scripts/preprocessing.py
# Apply median blur to sonar images and prepare Pascal VOC structure in line2voc_preprocessed/

import os
import cv2
import shutil
from pathlib import Path

# Define source and destination folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_IMG_DIR = PROJECT_ROOT / "data" / "line2voc" / "JPEGImages"
SRC_ANN_DIR = PROJECT_ROOT / "data" / "line2voc" / "Annotations"
SRC_SPLIT_DIR = PROJECT_ROOT / "data" / "line2voc" / "ImageSets" / "Main"

DEST_ROOT = PROJECT_ROOT / "data" / "line2voc_preprocessed"
DEST_IMG_DIR = DEST_ROOT / "JPEGImages"
DEST_ANN_DIR = DEST_ROOT / "Annotations"
DEST_SPLIT_DIR = DEST_ROOT / "ImageSets" / "Main"

# Create output folders
DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
DEST_ANN_DIR.mkdir(parents=True, exist_ok=True)
DEST_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Apply median blur to each image
count = 0
for img_path in SRC_IMG_DIR.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipped unreadable image: {img_path}")
        continue

    # Apply median filter with kernel size 3
    denoised = cv2.medianBlur(img, 3)

    # Save to destination folder
    out_path = DEST_IMG_DIR / img_path.name
    cv2.imwrite(str(out_path), denoised)
    count += 1

print(f"{count} images denoised and saved to: {DEST_IMG_DIR}")

# Copy annotation files
for xml_file in SRC_ANN_DIR.glob("*.xml"):
    shutil.copy(xml_file, DEST_ANN_DIR)

# Copy split files (train.txt, val.txt, test.txt)
for split_file in SRC_SPLIT_DIR.glob("*.txt"):
    shutil.copy(split_file, DEST_SPLIT_DIR)

print("Annotations and split files copied.")
print(f"Preprocessed dataset ready at: {DEST_ROOT}")