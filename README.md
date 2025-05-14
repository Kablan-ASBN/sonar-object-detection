# Sonar Object Detection using Faster R-CNN

This project is part of my MSc dissertation on object detection from sidescan sonar imagery using transfer learning

## Dataset

- `line2yolo/`: Raw dataset in YOLO format
- `line2voc/`: Converted Pascal VOC dataset with XML annotations
- `ImageSets/Main/`: Train/Val/Test splits

## Model

- Baseline: Faster R-CNN with ResNet50-FPN (PyTorch)
- Trained on raw sonar images (1,788 with objects)

## Model weight .pth
Model weights (.pth) are saved in a separate location in consideration of GitHub's file size restrictions.

Please access them via the following link: https://drive.google.com/file/d/1CMuWcLI2Dzaov8bNr2oY2SDNTOyIR-ug/view?usp=sharing


## Evaluation:
The Baseline model performance was tested using evaluate_model.py, which works by using the validation split which was defined in:
line2voc/ImageSets/Main/val.txt

The model underwent training on raw sidescan sonar images employing Faster R-CNN (ResNet50-FPN) from torchvision. Evaluation metrics were calculated using the MeanAveragePrecision class from TorchMetrics.

| Metric          | Value      | Description                                           |
| --------------- | ---------- | ----------------------------------------------------- |
| `mAP`           | **0.0441** | Mean Average Precision across IoU thresholds          |
| `mAP@0.50`      | **0.1643** | Precision at 50% IoU — basic object presence          |
| `mAP@0.75`      | 0.0109     | Strict localization performance (75% IoU)             |
| `mAP (small)`   | 0.0299     | Accuracy on small sonar objects                       |
| `mAP (medium)`  | 0.0774     | Accuracy on medium-sized targets                      |
| `mAP (large)`   | 0.1182     | Best results on large objects                         |
| `mAR@1`         | 0.0141     | Recall when allowing only top 1 prediction per image  |
| `mAR@10`        | 0.0687     | Recall for top 10 predictions                         |
| `mAR@100`       | **0.1623** | Max recall when evaluating top 100 predictions        |
| `mAR (small)`   | 0.1242     | Recall for small object detection                     |
| `mAR (medium)`  | 0.2482     | Recall for medium objects                             |
| `mAR (large)`   | 0.2286     | Recall for large object detection                     |
| `mAP per class` | -1.0000    | Not available — not supported in TorchMetrics summary |
| `mAR per class` | -1.0000    | Not available — not supported in TorchMetrics summary |

## Insight:
The model exhibited superior performance with large and medium-sized sonar objects. Detection of small targets is currently constrained and will be reexamined during the preprocessing and domain adaptation phases.