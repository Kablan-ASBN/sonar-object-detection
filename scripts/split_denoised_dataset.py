import os
import shutil
import random

# base dataset (already denoised)
BASE_DIR = "data/line2voc_preprocessed"

# output folders
SOURCE_DIR = "data/line2voc_source"
TARGET_DIR = "data/line2voc_target"

# reproducibility
random.seed(42)

# create output folders if they don’t exist
for folder in [SOURCE_DIR, TARGET_DIR]:
    os.makedirs(os.path.join(folder, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(folder, "Annotations"), exist_ok=True)
    os.makedirs(os.path.join(folder, "ImageSets", "Main"), exist_ok=True)

# get all image filenames (without extension)
image_names = [f.replace(".jpg", "") for f in os.listdir(os.path.join(BASE_DIR, "JPEGImages")) if f.endswith(".jpg")]
random.shuffle(image_names)

# simple 50/50 domain split
split_point = len(image_names) // 2
source_set = image_names[:split_point]
target_set = image_names[split_point:]

# copy files and create train.txt
def move_files(filenames, destination):
    for name in filenames:
        shutil.copy(os.path.join(BASE_DIR, "JPEGImages", f"{name}.jpg"),
                    os.path.join(destination, "JPEGImages", f"{name}.jpg"))
        shutil.copy(os.path.join(BASE_DIR, "Annotations", f"{name}.xml"),
                    os.path.join(destination, "Annotations", f"{name}.xml"))
        with open(os.path.join(destination, "ImageSets", "Main", "train.txt"), "a") as txt:
            txt.write(name + "\n")

# execute the split
move_files(source_set, SOURCE_DIR)
move_files(target_set, TARGET_DIR)

print("The Dataset split is completed: created line2voc_source and line2voc_target from preprocessed set.")
