# scripts/inference_and_visualization.py
# batch inference on all raw sonar images using my trained baseline Faster R-CNN model

import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# set project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# paths to raw images and trained baseline model
IMG_DIR = PROJECT_ROOT / "data" / "line2voc" / "JPEGImages"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vis"
PRED_CSV = PROJECT_ROOT / "outputs" / "preds.csv"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "baseline_sonar_fasterrcnn.pth"

# create folders if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and weights
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# hold predictions for CSV
all_preds = []

# loop through all sonar images
for img_path in IMG_DIR.glob("*.jpg"):
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = ToTensor()(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()
    labels = output["labels"].cpu()

    # convert to OpenCV format
    img_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    for i in range(len(boxes)):
        score = scores[i].item()
        if score < 0.5:
            continue

        x1, y1, x2, y2 = boxes[i].int().tolist()
        class_id = labels[i].item()

        # add to prediction list
        all_preds.append({
            "filename": img_path.name,
            "class_id": class_id,
            "score": score,
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
        })

        # draw box and score on image
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # save visual image with detections
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), img_np)

# save predictions as CSV
df = pd.DataFrame(all_preds)
df.to_csv(PRED_CSV, index=False)

print(f"Inference done. Visuals saved to: {OUTPUT_DIR}")
print(f"CSV with predictions saved to: {PRED_CSV}")