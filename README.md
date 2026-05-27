# Dual-Stage Deep Learning Framework for Breast Ultrasound Image Segmentation and Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Journal](https://img.shields.io/badge/Journal%20of%20Medical%20Systems-2025-green.svg)](https://doi.org/10.1007/s10916-025-02298-6)

Official implementation of the paper:

> **A Dual-stage Deep Learning Framework for Breast Ultrasound Image Segmentation and Classification**  
> Pierangela Bruno, Megan Macrì, Carmine Dodaro  
> *Journal of Medical Systems*, 2025 — [https://doi.org/10.1007/s10916-025-02298-6](https://doi.org/10.1007/s10916-025-02298-6)

---

## Overview

This repository provides a **modular, two-stage deep learning pipeline** for automated breast cancer diagnosis from ultrasound images:

1. **Stage 1 — Segmentation**: localises suspicious masses and produces a binary mask.
2. **Stage 2 — Classification**: classifies the cropped Region of Interest (ROI) as *benign* or *malignant*.

The two stages are explicitly decoupled but integrated in a single workflow, which allows each component to be swapped, upgraded, or validated independently — a key advantage over end-to-end multi-task approaches when working in clinical settings.

![Pipeline overview](assets/pipeline.png)
> *Input image → Segmentation Net → ROI identification → Cropping → Classification Net → Benign / Malignant*

---

## Key Results

| Method | Scenario | Seg Acc | IoU | Dice | Cls Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Our Approach** | A | **0.958** | 0.625 | **0.769** | **0.952** | **0.966** | 0.924 | **0.944** | **0.990** |
| **Our Approach** | B | 0.964 | **0.650** | **0.788** | **0.947** | 0.927 | 0.927 | **0.927** | **0.972** |
| **Our Approach** | C | **0.962** | **0.602** | **0.738** | **0.802** | **0.833** | **0.761** | **0.795** | **0.876** |
| Mask R-CNN | A | 0.957 | 0.691 | 0.762 | 0.809 | 0.768 | 0.744 | 0.754 | 0.744 |
| Mask R-CNN | C | 0.937 | 0.449 | 0.523 | 0.655 | 0.457 | 0.430 | 0.443 | 0.668 |
| Multi-task Transformers | A | 0.953 | 0.512 | 0.617 | 0.898 | 0.944 | 0.654 | 0.773 | 0.923 |

**Scenarios**: A = BUSI only · B = BUSI + USG combined · C = train on BUSI, test on USG (cross-domain)

---

## Method

### Segmentation stage

Encoder-decoder architectures are instantiated via [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch):

| Architecture | Encoder | Base channels | Dice (A) | IoU (A) |
|---|---|---|---|---|
| **DeepLabV3+** | **ResNet34** | **64** | **0.769** | **0.625** |
| U-Net | ResNet34 | 32 | 0.752 | 0.603 |
| DeepLabV3+ | ResNet34 | 32 | 0.747 | 0.596 |
| U-Net | EfficientNet-B0 | 32 | 0.734 | 0.580 |
| FPN | ResNet34 | 32 | — | — |

Training uses a composite loss: **0.4 × BCE + 0.5 × Dice + 0.5 × IoU**, Adam optimiser (lr=1e-4), ReduceLROnPlateau (factor=0.1, patience=3), early stopping (patience=10).

### Classification stage

The classifier operates solely on the cropped ROI. A pretrained backbone is frozen and only its last *K* parameter groups are unfrozen:

| Backbone | Head | Unfreeze K | Dropout | AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| **MobileNetV3-Small** | **MLP (128,64)** | **10** | **0.5** | **0.990** | **0.944** | **0.966** | 0.924 |
| EfficientNet-B0 | Linear | 10 | 0.0 | **0.990** | 0.941 | 0.935 | **0.947** |
| ResNet50 | Linear | 10 | 0.0 | 0.988 | 0.934 | 0.944 | 0.924 |
| DenseNet121 | MLP (128,64) | 10 | 0.5 | 0.988 | 0.945 | 0.956 | 0.935 |
| ResNet18 | MLP (128,64) | 10 | 0.5 | 0.988 | 0.911 | 0.932 | 0.891 |

Training uses BCEWithLogitsLoss, Adam (lr=1e-3), ReduceLROnPlateau, early stopping (patience=10).

---

## Repository structure

```
.
├── dual-stage-framework-complete.ipynb   # Main notebook (all 3 scenarios)
├── README.md
└── assets/
    └── pipeline.png                      # Pipeline figure
```

---

## Datasets

Two publicly available breast ultrasound datasets are used.

**BUSI** ([Al-Dhabyani et al., 2020](https://doi.org/10.1016/j.dib.2019.104863))
- 780 images from ~600 patients (Baheya Hospital, Egypt)
- 487 benign · 210 malignant · 83 normal
- 697 images retained (benign + malignant only)
- Available on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

**Breast-Lesions-USG** ([Pawłowska et al., 2024](https://doi.org/10.1038/s41597-024-02984-z))
- 256 patients, 266 annotated lesions (Cancer Imaging Archive)
- Acquired with various ultrasound devices → tests cross-dataset generalisation
- Available on [Kaggle](https://www.kaggle.com/datasets/vuppalaadithyasairam/breast-lesion-usg)

All images are resized to **256×256 grayscale**. Augmentation (rotations ±20°, translations ±20%, zoom, horizontal flip) is applied **only to training splits**, after patient-level stratified splitting, to prevent data leakage.

---

## Installation

```bash
git clone https://github.com/<your-username>/dual-stage-breast-us.git
cd dual-stage-breast-us

pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install scikit-learn imbalanced-learn pandas matplotlib pillow
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Running on Kaggle (recommended)

The notebook is designed to run on Kaggle with GPU acceleration. Add both datasets to your notebook environment and update the path constants at the top of the notebook if needed:

```python
PATH_BUSI = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT'
PATH_USG  = '/kaggle/input/breast-lesion-usg/BrEaST-Lesions_USG-images_and_masks'
```

---

## Usage

Open `dual-stage-framework-complete.ipynb` and run cells in order. The notebook is self-contained and structured in clearly labelled sections:

1. **Install & imports** — installs `segmentation-models-pytorch` and other dependencies
2. **Data loading** — loads both datasets, binarises masks, filters to benign/malignant only
3. **Augmentation & Dataset classes** — Albumentations pipeline, PyTorch `Dataset` wrappers
4. **Loss functions** — composite segmentation loss, IoU/Dice/pixel-accuracy metrics
5. **Segmentation model factory** — builds any `arch × encoder × base_channels` combination
6. **Classification model factory** — builds any backbone with configurable head and unfreeze depth
7. **ROI cropping helper** — deterministic bounding-box crop from predicted mask
8. **Scenario A** — architecture sweep on BUSI; produces Tables 1 & 2 equivalents
9. **Scenario B** — best config on combined BUSI + USG dataset
10. **Scenario C** — cross-domain evaluation: train BUSI → test USG
11. **Summary table** — Table 3 equivalent with all three scenarios
12. **Visualisations** — best/worst segmentation cases, confusion matrix

---

## Experimental scenarios

| Scenario | Train / Val | Test | Purpose |
|---|---|---|---|
| **A** | BUSI | BUSI | Architecture search & ablation |
| **B** | BUSI + USG | BUSI + USG | Robustness with additional data |
| **C** | BUSI | USG | Cross-dataset generalisation |

Data splits follow a **50-15-35** (train-val-test) ratio at the **patient level** to prevent leakage.

---

## Citation

If you use this code or the results in your work, please cite:

```bibtex
@article{bruno2025dualstage,
  title     = {A Dual-stage Deep Learning Framework for Breast Ultrasound
               Image Segmentation and Classification},
  author    = {Bruno, Pierangela and Macrì, Megan and Dodaro, Carmine},
  journal   = {Journal of Medical Systems},
  volume    = {49},
  pages     = {162},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s10916-025-02298-6}
}
```

---

## Funding

This work was supported by the European Union – NextGenerationEU and the Italian Ministry of Research (MUR) under PNRR projects **FAIR** (CUP H23C22000860006) and **Tech4You** (CUP H23C22000370006). Carmine Dodaro was also supported by MUR/NRRP project **RAISE** (ECS00000035), sub-project GOLD (CUP H53C24000400006).

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).