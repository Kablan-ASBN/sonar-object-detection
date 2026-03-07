# Sonar Object Detection with Transfer Learning and Domain Adaptation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-Detection%20API-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=flat)
![Context](https://img.shields.io/badge/Context-MSc%20Research%20Placement-6c757d?style=flat)

## Context

This project originated from a professional placement at [Seabed.AI](https://seabed.ai), a UK-based startup specialising in autonomous seabed survey and mapping. The research conducted during that placement became the foundation of my MSc Data Science dissertation at the University of Greenwich.

The core contribution is **DCCAN (Domain-Conditional Combined Adversarial Network)**, a novel domain adaptation architecture I designed and implemented to address sonar object detection under domain shift. DCCAN is not an existing model from the literature — it was proposed, built, and evaluated as original work during this project.

This repository documents the core technical work. Day-to-day responsibilities at Seabed.AI also included data validation, ETL support, dashboarding, and cross-team reporting, which are not fully represented here due to NDA and scope.

## Problem Framing

Sidescan sonar is critical for seabed mapping, infrastructure inspection, archaeology, and defense. Automatic detection is difficult because images contain speckle noise, low contrast, and strong variation between surveys, and labeled datasets are scarce. The core challenge addressed was improving object detection robustness when training and deployment data differ significantly due to these environmental and sensor factors.

## Emphasis

- Data preparation and validation under imperfect conditions
- Evaluation discipline and precision-recall trade-offs
- Robustness and stability rather than benchmark performance

**Author:** Gomis Kablan Assebian

## Repository Structure

```
sonar-object-detection/
├── scripts/
│   ├── convert_yolo_to_voc.ipynb    # YOLO → Pascal VOC conversion pipeline
│   ├── voc_dataset.py               # Custom PyTorch Dataset for VOC-format sonar data
│   └── loss_wrappers.py             # AMP-safe domain adversarial loss wrapper
├── notebooks/
│   ├── Faster_RCNN_Baseline_Model.ipynb          # Baseline training (raw/denoised/CLAHE)
│   ├── Faster_RCNN_Baseline_Model_20_Epoch.ipynb # 20-epoch baseline experiments
│   └── DANN_DCCAN_20_25_epoch_tuned.ipynb        # DANN + DCCAN training & evaluation
├── outputs/                         # Detection visualizations and evaluation plots
│   ├── detection_per_image_20ep_mod.png
│   └── distri_of_detect_20_ep_mod.png
└── README.md
```

## Dataset

- **Source**: 3,465 sidescan sonar images annotated in YOLO format
- **Classes**:
  - `object` (class 1): physical structures on the seabed
  - `shadow` (class 2): acoustic shadows behind objects
  - `background` (class 0): implicit class

### Conversion

YOLO labels were converted to **Pascal VOC** format using `scripts/convert_yolo_to_voc.ipynb`. The pipeline uses multi-threaded batch processing to convert bounding boxes to VOC coordinates, filter empty or malformed annotations, remap class labels (correcting an early mislabeling issue in the original scheme), and generate the standard VOC directory structure (`Annotations/`, `JPEGImages/`, `ImageSets/Main/`). A stratified split (80/10/10 train/val/test) preserves class balance across subsets.

## Preprocessing Variants

Three dataset configurations were constructed to reflect different domain characteristics:

- **Raw**: Unmodified sonar images — the target domain, reflecting full operational noise and variability.
- **Denoised**: 3x3 median blur for speckle noise reduction, used as the labeled source domain in adaptation experiments.
- **CLAHE + Augmented**: Contrast-limited adaptive histogram equalization followed by geometric augmentations (flips, scaling, brightness jitter). Visually sharper, but amplified background textures that increased false positives, degrading overall detector precision.

## Models

**Detector**: Faster R-CNN with ResNet-50 FPN backbone (pretrained on COCO), with the prediction head replaced for 3 classes.

### Variants

- **Baselines**: trained on raw, denoised, or CLAHE-augmented sonar data
- **DANN**: Domain-Adversarial Neural Network with global feature alignment via Gradient Reversal Layer
- **DCCAN**: Domain-Conditional Combined Adversarial Network (hybrid adversarial model, proposed in this work)

## DCCAN: A Novel Model Proposed in This Work

**Problem:**
Standard transfer learning struggled with sonar domain shift. Baselines plateaued at ~0.15 AP50. DANN improved recall but hurt precision, while standalone CDAN (class-conditional alignment) was numerically unstable under automatic mixed precision due to its outer-product mapping.

**Idea:**
Combine the strengths of DANN and CDAN while adding a third alignment path to stabilize learning. The design was motivated by observing that shadows vs. objects require class-aware alignment (CDAN), global differences between raw and denoised sonar require holistic alignment (DANN), and early region proposals carry domain cues that benefit from low-level alignment.

**Architecture:**
DCCAN integrates three adversarial components into Faster R-CNN, each attached via a Gradient Reversal Layer:

1. **Global alignment (DANN-style):** A domain discriminator (MLP, 1024 hidden units) operates on adaptive-average-pooled backbone features (256-d).

2. **Class-conditional alignment (CDAN-style):** A discriminator receives the flattened outer product of L2-normalized pooled features and proxy class logits (256 x 3 = 768-d). Stabilized with temperature sharpening (T = 0.6) and confidence gating that filters low-certainty samples early in training (threshold decays from 0.40 to 0.20 over training).

3. **Proposal-level alignment:** A lightweight 2-layer convolutional domain head (Conv 3x3 → ReLU → Conv 1x1) operates directly on FPN P3 feature maps, with global average pooling to produce per-sample domain logits.

**Training objective:**

```
L_total = L_det + λ_DANN(t) · L_DANN + λ_CDAN(t) · L_CDAN + λ_RPN(t) · L_RPN
```

- `L_det`: standard Faster R-CNN detection loss (classification + regression) on labeled source data
- `L_DANN`, `L_CDAN`, `L_RPN`: binary cross-entropy domain losses from the three alignment paths
- Each λ follows a logistic ramp schedule: `γ(p) = 2 / (1 + exp(−10p)) − 1`, where `p ∈ [0,1]` is normalized training progress
- Fixed maximum coefficients: **λ_DANN = 0.20**, **λ_CDAN = 0.30**, **λ_RPN = 0.10**

**Stability safeguards:** 2-epoch backbone freeze warmup (detection head only, no adaptation pressure), gradient clipping (max norm 5.0), and all training under AMP (autocast + GradScaler).

**Result:**
A stable, mixed-precision compatible model that achieved the best overall precision-recall balance across all experiments.

## Training

- **Framework**: PyTorch + TorchVision detection API
- **Hardware**: NVIDIA A100 GPU (Google Colab Pro)
- **Precision**: AMP enabled for all runs
- **Epochs**: 20 for all models

### Baselines

- **Optimizer**: AdamW (lr = 2x10⁻⁴, weight decay = 1x10⁻⁴)
- **Scheduler**: CosineAnnealingLR (T_max = 20, eta_min = 1x10⁻⁶)
- **Batch size**: 8

### DANN

- **Optimizer**: SGD (detector lr = 1x10⁻³, discriminator lr = 2x10⁻³, momentum = 0.9, weight decay = 5x10⁻⁴)
- **Batch size**: 12 (source + target)
- **GRL ramp**: logistic schedule, max coefficient = 0.20

### DCCAN

- **Optimizer**: SGD with separate parameter groups:
  - Detector + proxy classifier: lr = 1.5x10⁻³
  - Discriminators: lr = 4.5x10⁻⁴
  - Both: momentum = 0.9, weight decay = 5x10⁻⁴
- **Batch size**: 12 (source + target)
- **Backbone freeze**: first 2 epochs
- **Gradient clipping**: max norm = 5.0

## Results

### Final Evaluation (Raw Target Domain — Validation Split)

| Model Variant | AP50 | mAP@[.50:.95] | mAR@100 | Notes |
|---|---|---|---|---|
| Baseline (Raw) | 0.149 | 0.039 | 0.104 | Shadows easier to detect than objects |
| Baseline (Denoised) | 0.152 | 0.039 | 0.107 | Slight stability improvement |
| Baseline (CLAHE + Aug) | 0.117 | 0.035 | 0.088 | Background amplification hurt precision |
| DANN | 0.091 | 0.024 | 0.115 | Recall improved, precision dropped |
| **DCCAN** | **0.163** | **0.043** | **0.155** | Best overall balance |

### FROC Analysis

FROC (Free-Response ROC) curves complement COCO metrics by showing how recall changes as the false-positive allowance per image (FPPI) increases.

- **Baselines**: strongest at low FPPI (~10-15), reaching recall ~0.30-0.31, conservative but saturating quickly.
- **DANN**: extends recall but at steep precision cost, plateauing below DCCAN.
- **DCCAN**: consistently dominates at higher FPPI, rising to nearly 0.47 recall, recovering significantly more true targets when moderate false-positive rates are acceptable.

This positions DCCAN as the most effective model in settings where maximizing target coverage matters more than strict conservatism (e.g., mine countermeasures, infrastructure inspection).

## Visualizations

![Detection Per Image](outputs/detection_per_image_20ep_mod.png)

Per-image detection count boxplots across all model variants. Shows how detection volume and variance differ between baseline, DANN, and DCCAN configurations.

![Detection Distribution](outputs/distri_of_detect_20_ep_mod.png)

Detection count distribution histograms with KDE overlays. Illustrates the spread and concentration of detections across the validation set for each model.

## Key Skills Demonstrated

- **Data engineering**: YOLO to VOC format conversion, label remapping, stratified splits, multi-threaded batch preprocessing
- **Deep learning**: Faster R-CNN, adversarial domain adaptation (DANN, CDAN, hybrid DCCAN), PyTorch/TorchVision APIs
- **Transfer learning**: COCO-pretrained backbones adapted to sonar imagery
- **Domain adaptation**: implemented DANN baseline and novel DCCAN hybrid with three alignment paths
- **Evaluation**: COCO metrics (TorchMetrics) and FROC analysis for precision-recall characterization
- **Reproducibility**: modular code, notebooks, saved checkpoints, exported loss logs and CSVs

## Roadmap

- Investigate adaptive or learnable λ-weighting for the three adversarial losses
- Integrate pseudo-labeling and entropy minimization for unlabeled target domain
- Extend pipeline to additional sonar modalities (FLS, SAS, multibeam) and cross-survey evaluation
- Explore contrastive or self-supervised pretraining for backbone initialization
- Experiment with transformer-based or attention-heavy architectures for long-range object-shadow context
- Add explainability tools (class activation maps, embedding visualizations) for operational trust
- Optimize inference for embedded maritime hardware (AUV deployment)

## License

Research-only. Non-commercial use unless otherwise agreed.
For collaborations or applied deployments, please open an issue.

*Author: [Kablan Assebian](https://www.linkedin.com/in/gomis-kablan/) · [GitHub](https://github.com/Kablan-ASBN)*
