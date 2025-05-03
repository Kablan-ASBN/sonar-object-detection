# scripts/inference_and_visualization.py
# Step 1.7 — Batch inference on full JPEGImages folder using Seabed.ai's pretrained model

import os
import cv2
import csv
import pandas as pd
from pathlib import Path
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from load_model import load_model

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "source" / "JPEGImages"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vis"
PRED_CSV = PROJECT_ROOT / "outputs" / "preds.csv"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "model_0007959.pth"

# Ensure output folders exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

# Load Seabed.ai's pretrained Detectron2 model
predictor, cfg = load_model(str(MODEL_PATH))

# Set padded metadata to avoid IndexError when drawing predictions
MetadataCatalog.get("sonar_dataset").set(thing_classes=["object"] * 100)

# Collect all predictions here
all_preds = []

# Loop over all sonar images in JPEGImages folder
for img_path in IMG_DIR.glob("*.jpg"):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Could not read image: {img_path}")
        continue

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Extract predictions (if any)
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []

    # Store each prediction row into all_preds
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        class_id = int(classes[i])
        score = float(scores[i])

        all_preds.append({
            "filename": img_path.name,
            "class_id": class_id,
            "score": score,
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
        })

    # Visualize predictions and save images
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("sonar_dataset"), scale=1.2)
    out = v.draw_instance_predictions(instances)
    out_img = out.get_image()[:, :, ::-1]
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), out_img)

# Save predictions to CSV
df = pd.DataFrame(all_preds)
df.to_csv(PRED_CSV, index=False)

print(f"Inference complete. Visuals saved to: {OUTPUT_DIR}")
print(f"Predictions saved to CSV: {PRED_CSV}")
