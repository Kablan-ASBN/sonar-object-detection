## Sonar Object Detection using Transfer Learning and Domain Adaptation

This repository is part of my MSc dissertation titled:

**"Object Detection in Sidescan Sonar Data Using Transfer Learning and Domain Adaptation to Reduce the Need for Manual Annotation"**

Sidescan sonar images are often noisy and inconsistent across different environments. This makes it difficult for object detection models to generalize and also makes manual annotation costly and time-consuming. The goal of this project was to develop a pipeline that:

* Uses transfer learning to reduce training time and the need for large amounts of labeled data
* Applies preprocessing techniques to clean and enhance low-quality sonar images
* Evaluates how denoising and augmentation affect model performance
* Applies domain adaptation (DANN/CDAN) to improve generalization across different underwater terrains
* Optionally explores few-shot learning for future extension

All models were implemented using PyTorch and torchvision. The sonar dataset was converted to Pascal VOC format and used to train multiple versions of Faster R-CNN. All training, evaluation, visualization, and preprocessing scripts are structured for reproducibility.

> This project is conducted under NDA with Seabed.ai. All outputs and code comply with those restrictions.

---

### Dataset

* `line2yolo/`: Original dataset in YOLO format
* `line2voc/`: Converted Pascal VOC format (XML)
* `line2voc_preprocessed/`: Median-blurred version of the dataset
* `line2voc_preprocessed_augmented/`: CLAHE + offline augmentations (flip, rotate, jitter)
* `ImageSets/Main/`: Train, val, and test splits (identical across all dataset versions)

---

### Model Architecture

Used `fasterrcnn_resnet50_fpn` from `torchvision` with a custom classification head (3 classes):

* Model 1: Trained on raw sonar images (baseline)
* Model 2: Trained on median-blurred images
* Model 3: Trained on CLAHE + augmented images

---

### Training Configuration

* Pretrained COCO weights
* Optimizer: AdamW (LR = 1e-4)
* Epochs: 5
* Batch size: 2

**Training Scripts:**

* `train_faster_rcnn.py` (raw sonar)
* `train_denoised_fasterrcnn.py`
* `train_augmented_fasterrcnn.py`

---

### Checkpoints

Model weights are stored externally:

* **Raw (baseline)**:
  [Download](https://drive.google.com/file/d/1CMuWcLI2Dzaov8bNr2oY2SDNTOyIR-ug/view?usp=sharing)

* **Denoised model**:
  [Download](https://drive.google.com/file/d/1-F5k6tRJNg9JVDQv0NOtfnSoT3Dzaa_W/view?usp=sharing)

* **Augmented model**:
  [Download](https://drive.google.com/file/d/191dtnr4owKMqI9l2liCBFlkdwY2fVScC/view?usp=sharing)

---

### Evaluation: Raw Sonar Images (Baseline)

| Metric    | Value  |
| --------- | ------ |
| mAP       | 0.0441 |
| mAP\@0.50 | 0.1643 |
| mAP\@0.75 | 0.0109 |
| mAR\@100  | 0.1623 |

**Insight:**
The baseline model worked best on medium and large objects. Small object detection was limited, motivating the use of preprocessing.

---

### Evaluation: Denoised (Median Blur)

| Metric    | Value  |
| --------- | ------ |
| mAP       | 0.0475 |
| mAP\@0.50 | 0.1693 |
| mAP\@0.75 | 0.0143 |
| mAR\@100  | 0.1677 |

**Insight:**
Denoising slightly improved detection for small and medium objects. However, smoothing may have affected large object clarity.

---

### Evaluation: CLAHE + Augmented

| Metric    | Value  |
| --------- | ------ |
| mAP       | 0.0026 |
| mAP\@0.50 | 0.0084 |
| mAP\@0.75 | 0.0009 |
| mAR\@100  | 0.0272 |

**Insight:**
This model underperformed significantly. The offline augmentations likely caused misalignment between input images and bounding box annotations.

---

### Evaluation: Domain Adaptation with DANN

To improve generalization across different sonar domains, I used Domain-Adversarial Neural Networks (DANN) with the THUML Transfer Learning Library.

* Source domain: `line2voc_preprocessed` (denoised)
* Target domain: `line2voc` (raw sonar)
* Model: Faster R-CNN with ResNet-50 FPN
* Training: 5 epochs on Google Colab Pro (GPU)

---

#### DANN Results (score threshold = 0.5)

Notebook: `notebooks/DANN_Pipeline_Setup(0.5).ipynb`

| Epoch | Detection Loss | Domain Loss |
| ----- | -------------- | ----------- |
| 1     | 1035.85        | 583.76      |
| 2     | 988.75         | 596.74      |
| 3     | 966.12         | 595.68      |
| 4     | 943.11         | 595.41      |
| 5     | 934.16         | 603.28      |

| Metric    | Value  |
| --------- | ------ |
| mAP       | 0.0495 |
| mAP\@0.50 | 0.1796 |
| mAP\@0.75 | 0.0173 |
| mAR\@100  | 0.1529 |
| Precision | 0.0918 |
| Recall    | 0.4509 |

**Visualizations:**

#### Detection Summary (Score Threshold 0.5)

![Detection Histogram - Score 0.5](outputs/histo_dann_0.5.png)  
*Histogram of detections per image with DANN threshold = 0.5*

![Boxplot of Detections - Score 0.5](outputs/boxplot_dann_0.5.png)  
*Boxplot showing distribution of detections per image*

![Detection Table - Score 0.5](outputs/table_dann_0.5.png)  
*Summary statistics: baseline vs. DANN at threshold 0.5*


---

#### DANN Results (score threshold = 0.7, Final Model)

Notebook: `notebooks/DANN_Pipeline_Setup(0.7).ipynb`

| Epoch | Detection Loss | Domain Loss |
| ----- | -------------- | ----------- |
| 1     | 1030.13        | 708.96      |
| 2     | 977.82         | 720.08      |
| 3     | 955.54         | 727.78      |
| 4     | 937.11         | 728.70      |
| 5     | 926.48         | 727.61      |

| Metric    | Value  |
| --------- | ------ |
| mAP       | 0.0456 |
| mAP\@0.50 | 0.1653 |
| mAP\@0.75 | 0.0157 |
| mAR\@100  | 0.1543 |
| Precision | 0.0887 |
| Recall    | 0.4544 |

**Visualizations:**

#### Detection Summary (Score Threshold 0.7)

![Detection Histogram - Score 0.7](outputs/histo_dann_0.7.png)  
*Histogram of detections per image with DANN threshold = 0.7*

![Boxplot of Detections - Score 0.7](outputs/boxplot_dann_0.7.png)  
*Boxplot showing distribution of detections per image*

![Detection Table - Score 0.7](outputs/table_dann_0.7.png)  
*Summary statistics: baseline vs. DANN at threshold 0.7*


**Threshold Insight:**
A score threshold of 0.7 was ultimately selected to reduce noisy detections. While it lowers recall slightly, it significantly improves detection focus and suppresses over-prediction noise. This aligns with the project's aim to build a robust detector that is practical for real-world sonar interpretation and minimizes false alarms.

---

### Visual Outputs

* `outputs/vis/`: Raw baseline model predictions
* `outputs/vis_denoised/`: Median-filtered model predictions
* `outputs/vis_augmented/`: CLAHE + augmented model predictions
* `outputs/dann_vis/`: Domain-adapted predictions from DANN model

---

### Notebooks

* `notebooks/Faster_RCNN_Baseline_Model.ipynb`
* `notebooks/DANN_Pipeline_Setup(0.5).ipynb`
* `notebooks/DANN_Pipeline_Setup(0.7).ipynb`

These notebooks contain the complete training, evaluation, and visualization workflows. They are structured for reproducibility and ready to support both dissertation submission and viva presentation.

---