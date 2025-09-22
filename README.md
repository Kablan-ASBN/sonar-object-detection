# Sonar Object Detection with Transfer Learning and Domain Adaptation

This repository implements a full machine learning pipeline for **object detection in sidescan sonar imagery**.  
It explores how preprocessing, transfer learning, and domain adaptation can improve generalization under severe noise and domain shift, where training and deployment data differ significantly.

The system integrates:

- **Data engineering**: YOLO-to-VOC conversion, class remapping, stratified dataset splits  
- **Detection models**: Faster R-CNN with ResNet-50 FPN backbone  
- **Domain adaptation**: adversarial approaches (DANN, DCCAN) to reduce reliance on target labels  
- **Evaluation**: COCO-style metrics and Free-response ROC (FROC) curves  

**Author:** Gomis Kablan Assebian

---

## Dataset

- **Source**: 3,465 sidescan sonar images annotated in YOLO format  
- **Classes**:
  - `object` — physical structures  
  - `shadow` — acoustic shadows behind objects  
  - `background` — implicit class  

### Conversion
YOLO labels were converted to **Pascal VOC** format using `scripts/convert_yolo_to_voc.ipynb`:
- Converted bounding boxes to VOC coordinates  
- Filtered empty labels  
- Generated `Annotations/`, `JPEGImages/`, and `ImageSets/Main/`  

---

## Preprocessing Variants

- **Raw**: unmodified sonar images  
- **Denoised**: median blur for speckle noise reduction  
- **CLAHE + Augmented**: contrast enhancement and geometric augmentations  

> Note: CLAHE-augmented data degraded performance due to bounding box misalignments.

---

## Models

**Detector**: Faster R-CNN with ResNet-50 FPN backbone (pretrained on COCO).  

### Variants
- **Baseline** — trained on raw, denoised, or augmented sonar data  
- **DANN** — Domain-Adversarial Neural Network (global feature alignment with Gradient Reversal Layer)  
- **DCCAN** — Domain-Conditional Combined Adversarial Network (hybrid adversarial model)

---

## How DCCAN Works

**Problem:**  
Standard transfer learning struggled with sonar domain shift. Baselines plateaued at ~0.15 AP50.  
DANN improved recall but hurt precision, while CDAN (class-conditional alignment) was unstable under mixed precision.  

**Idea:**  
Combine the strengths of both DANN and CDAN while adding a third alignment path to stabilize learning.  
The inspiration came from observing:
- Shadows vs objects require **class-aware alignment** (CDAN).  
- Global differences between raw and denoised sonar require **holistic alignment** (DANN).  
- Early region proposals also carry domain cues that benefit from **low-level alignment**.  

**Architecture:**  
DCCAN integrates three adversarial components into Faster R-CNN:

1. **Global alignment (DANN-style):**  
   - A domain discriminator operates on pooled backbone features.  

2. **Class-conditional alignment (CDAN-style):**  
   - A discriminator receives the interaction of features × logits.  
   - Stabilized with **temperature sharpening** and **confidence gating** to filter low-certainty samples.  

3. **Proposal-level alignment:**  
   - A lightweight convolutional domain head aligns FPN proposal features.  

**Training objective:**  

L_total = L_det + λ_DANN * L_DANN + λ_CDAN * L_CDAN + λ_RPN * L_RPN
- `L_det`: detection loss (classification + regression)  
- `L_DANN`, `L_CDAN`, `L_RPN`: domain losses from the three alignment paths  
- Coefficients follow a **ramp schedule** that grows over training for stability  

**Result:**  
A stable, mixed-precision compatible model that improved recall and balanced the precision–recall trade-off.

---

## Training

- **Framework**: PyTorch + TorchVision detection API  
- **Hardware**: NVIDIA A100 GPU (Colab Pro)  
- **Precision**: AMP enabled for all 20-epoch runs  
- **Optimizer**: SGD  
  - LR = 0.005 (baselines), 0.001 (DA)  
  - Momentum = 0.9, weight decay = 5e-4  
- **Batch size**: 8 (baselines), 12 (DA models)  

---

## Results

### Final Evaluation (Raw Target Domain)

| Model Variant          | AP50   | mAP@[.50:.95] | Notes                                 |
|------------------------|--------|---------------|---------------------------------------|
| Baseline (Raw)         | 0.152  | 0.039         | Shadows easier to detect than objects |
| Baseline (Denoised)    | 0.158  | 0.041         | Slight boost in precision             |
| Baseline (CLAHE + Aug) | 0.019  | 0.008         | Over-detection, poor precision        |
| DANN                   | 0.093  | 0.025         | Recall improved, precision dropped    |
| **DCCAN**              | **0.163** | **0.043**  | Best overall balance                  |

### FROC Analysis
- Baselines: stronger at **low false-positive rates**  
- DCCAN: recovered more **true targets** at higher FPPI, showing better generalization under domain shift  

---

## Visualizations

Detection visualizations and statistics are stored in `outputs/`:
- Boxplots and histograms of detection counts  
- Bounding box overlays for each model (Raw, Denoised, CLAHE, DANN, DCCAN)  



---

## Key Skills Demonstrated

- **Data engineering**: format conversion, label remapping, stratified splits, preprocessing  
- **Deep learning**: Faster R-CNN, adversarial domain adaptation, PyTorch/TorchVision APIs  
- **Transfer learning**: COCO-pretrained backbones adapted to sonar  
- **Domain adaptation**: implemented DANN + novel DCCAN hybrid  
- **Evaluation**: COCO metrics and FROC analysis  
- **Reproducibility**: modular code, notebooks, and scripts  

---

## Roadmap

- Extend pipeline to additional sonar modalities (FLS, SAS, multibeam)  
- Explore transformer-based backbones and contrastive pretraining  
- Integrate pseudo-labeling and entropy minimization for target domain  
- Optimize inference for embedded maritime hardware  

---

## License

Research-only. Non-commercial use unless otherwise agreed.  
For collaborations or applied deployments, please open an issue.
