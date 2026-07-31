# CAMAD: Class-Aware Multi-Dimensional Framework for Imbalanced Skin Lesion Classification

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Authors
- **Mati Nakphon** (Lead) — University of Europe for Applied Sciences
- **Sarawut Boonyarat** — University of Europe for Applied Sciences

## Overview

CAMAD addresses class imbalance in dermoscopic skin lesion classification at three levels:

1. **Data level** — class-specific augmentation + inverse-frequency balanced sampling
2. **Algorithmic level** — Focal Loss (γ = 2.0)
3. **Architectural level** — CBAM attention modules after each ResNet-50 residual stage

A clinical threshold (τ = 0.20) is selected on validation to maximise melanoma sensitivity while maintaining specificity ≥ 0.80.

## Dataset

- **HAM10000**: 10,015 dermoscopic images, 7 classes (NV 67%, MEL 11%)
- **Lesion-level stratified split**: 70/15/15 (1,497 test images, 168 melanomas)

## Key Results

| Metric | CAMAD (τ=0.20) |
| :--- | :---: |
| Macro F1 | 0.6889 |
| Melanoma recall | **86.31%** |
| Malignant avg recall | 77.42% |
| Benign avg precision | 85.59% |
| Melanoma → Nevus errors | **16 / 168** (vs 55 for ResNet-50) |

## Getting Started

```bash
git clone https://github.com/oagaudit/CAMAD.git
cd CAMAD
pip install -r requirements.txt
