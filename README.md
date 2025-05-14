# Sonar Object Detection using Transfer Learning and Domain Adaptation

This repository is part of my MSc dissertation titled:

**"Object Detection in Sidescan Sonar Data Using Transfer Learning and Domain Adaptation to Reduce the Need for Manual Annotation"**

Sidescan sonar images are noisy and variable across environments, making it hard for object detection models to generalize. Labeling this data is also time-consuming and expensive. The goal of this project was to build a sonar object detection pipeline that:

- Uses **transfer learning** to reduce training time and labeled data needs
- Applies **preprocessing** to clean and enhance noisy sonar imagery
- Evaluates how denoising and augmentation affect model performance
- Prepares for **domain adaptation** (DANN/CDAN) to generalize across new sonar textures
- Optionally considers few-shot learning for future extension

I implemented all models using PyTorch and torchvision. My dataset was converted to Pascal VOC format and used to train multiple versions of Faster R-CNN. All core training, evaluation, visualization, and preprocessing scripts are organized and reproducible. The results will be used to evaluate model generalization across noisy and domain-shifted sonar data.

This work is being conducted under NDA with Seabed.ai. All code and outputs remain within the constraints of that agreement.


---

## Dataset

- `line2yolo/`: Original dataset in YOLO format
- `line2voc/`: Converted version in Pascal VOC format (XML)
- `line2voc_preprocessed/`: Median-blurred images
- `line2voc_preprocessed_augmented/`: CLAHE + offline augmentations (flip, rotate, jitter)
- `ImageSets/Main/`: Train, val, and test splits (same across all datasets)

---

## Model Architecture

I used `fasterrcnn_resnet50_fpn` from torchvision with a custom head for 3 classes (background, object, object_alt). The model was trained 3 times:

1. On raw sonar images (baseline)
2. On denoised images (median filter)
3. On CLAHE + augmented images

---

## Training

- All models used pretrained COCO weights
- Optimizer: AdamW, LR = 1e-4
- Epochs: 5
- Batch size = 2

Training scripts:
- `models/train_faster_rcnn.py` (raw)
- `models/train_denoised_fasterrcnn.py`
- `models/train_augmented_fasterrcnn.py`

---

## Checkpoints

Model weights are stored externally:

- **Raw (baseline)**:  
  [Download](https://drive.google.com/file/d/1CMuWcLI2Dzaov8bNr2oY2SDNTOyIR-ug/view?usp=sharing)

- **Denoised model**:  
  [Download](https://drive.google.com/file/d/1-F5k6tRJNg9JVDQv0NOtfnSoT3Dzaa_W/view?usp=sharing)

- **Augmented model**:  
  [Download](https://drive.google.com/file/d/191dtnr4owKMqI9l2liCBFlkdwY2fVScC/view?usp=sharing)

---

## Evaluation: Raw Sonar Images (Baseline)

| Metric          | Value      |
|-----------------|------------|
| `mAP`           | 0.0441     |
| `mAP@0.50`      | 0.1643     |
| `mAP@0.75`      | 0.0109     |
| `mAR@100`       | 0.1623     |

**Insight:**  
The baseline model performed best on medium and large sonar objects. Performance on small targets was limited, so I tested preprocessing to improve generalization.

---

## Evaluation: Denoised Model (Median Blur)

| Metric          | Value      |
|-----------------|------------|
| `mAP`           | 0.0475     |
| `mAP@0.50`      | 0.1693     |
| `mAP@0.75`      | 0.0143     |
| `mAR@100`       | 0.1677     |

**Insight:**  
Denoising improved detection of small and medium targets slightly. mAP improved, but large-object detection dropped, probably due to smoother texture.

---

## Evaluation: CLAHE + Augmented Images

| Metric          | Value      |
|-----------------|------------|
| `mAP`           | 0.0026     |
| `mAP@0.50`      | 0.0084     |
| `mAP@0.75`      | 0.0009     |
| `mAR@100`       | 0.0272     |

**Insight:**  
This version underperformed across all metrics. CLAHE + offline augmentations (flips, rotation, contrast jitter) likely introduced too much variation or label misalignment. The model struggled to generalize.

---

## Final Model Comparison

| Model Type      | mAP    | mAP@50 | mAR@100 |
|-----------------|--------|--------|---------|
| Raw (baseline)  | 0.0441 | 0.1643 | 0.1623  |
| Denoised        | 0.0475 | 0.1693 | 0.1677  |
| Augmented       | 0.0026 | 0.0084 | 0.0272  |

**Conclusion:**  
Denoising alone gave the best results overall. CLAHE + augmentations hurt performance, suggesting augmentation should be done on-the-fly during training, not saved offline.

---

## Visual Outputs

- `outputs/vis/` – raw model predictions
- `outputs/vis_denoised/` – denoised predictions
- `outputs/vis_augmented/` – augmented model predictions

Predictions were generated using:
- `visualize_predictions.py`
- `visualize_predictions_denoised.py`
- `visualize_predictions_augmented.py`

---

## Notebook

Everything was tested in:  
`notebooks/Faster_RCNN_Baseline_Model.ipynb`

This notebook contains:

- All training, evaluation, and visualization steps
- All metrics and visual examples
- Ready for submission and Viva walkthrough