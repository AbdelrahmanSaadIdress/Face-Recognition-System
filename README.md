<div align="center">

# 🧠 Face Recognition System
### A Comparative Study of Deep Face Recognition Loss Functions

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Active%20Research-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

*Implementing, training, and benchmarking three landmark face recognition loss functions — ArcFace, Triplet Loss, and SphereFace — toward building a production-ready attendance system.*

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Research Papers](#research-papers)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Experiments](#experiments)
  - [Round 1 — Initial Training & Baseline Results](#round-1--initial-training--baseline-results)
  - [Round 2 — Hyperparameter Tuning & Advanced Analysis](#round-2--hyperparameter-tuning--advanced-analysis-coming-soon)
- [Final Model Selection](#final-model-selection)
- [Project Roadmap](#project-roadmap)
- [Installation & Usage](#installation--usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview

This repository is a systematic research effort to implement and evaluate three of the most influential deep face recognition methodologies from scratch:

| Method | Loss Type | Core Idea |
|---|---|---|
| **Triplet Loss** | Metric Learning | Directly optimizes embedding distances via anchor-positive-negative triplets |
| **SphereFace** | Multiplicative Angular Margin | Deep hypersphere embedding with multiplicative angular penalty |
| **ArcFace** | Angular Margin | Additive angular margin in the hyperspherical embedding space |


The ultimate goal is to select the best-performing approach for integration into a **real-world attendance system**, where identity verification must be fast, accurate, and robust to real-world variation (lighting, pose, expression).

---

## Motivation

Classical softmax-based classifiers struggle with the open-set face verification problem — recognizing identities never seen during training. Metric learning and margin-based loss functions address this by learning embedding spaces with strong intra-class compactness and inter-class separability. This project compares these three paradigms under controlled, identical training conditions to determine which is best suited for a deployment-ready attendance system.

---

## Research Papers

The three methods implemented in this project are grounded in the following seminal publications:

| # | Paper | Authors | Venue | Link |
|---|---|---|---|---|
| 1 | **FaceNet: A Unified Embedding for Face Recognition and Clustering** *(Triplet Loss)* | Schroff et al. | CVPR 2015 | [📄 arXiv:1503.03832](https://arxiv.org/abs/1503.03832) |
| 2 | **SphereFace: Deep Hypersphere Embedding for Face Recognition** | Liu et al. | CVPR 2017 | [📄 arXiv:1704.08063](https://arxiv.org/abs/1704.08063) |
| 3 | **ArcFace: Additive Angular Margin Loss for Deep Face Recognition** | Deng et al. | CVPR 2019 | [📄 arXiv:1801.07698](https://arxiv.org/abs/1801.07698) |



---

## Datasets

### 🏋️ Training & Validation — VGGFace2 (112×112)

> **Dataset:** [VGGFace2 112×112 — Kaggle](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112)

A large-scale face dataset with images captured in the wild, covering wide variations in pose, age, illumination, and ethnicity. All images are pre-aligned and cropped to 112×112 pixels.

| Property | Value |
|---|---|
| Subjects | ~9,000 identities |
| Images | ~3.3 million |
| Resolution | 112 × 112 |
| Split Used | Train / Validation |

⚠️ **Important Note on Data Usage**

Due to the large scale of VGGFace2 (~3.3M images), the full dataset was **not used in all experiments**. Instead, a **controlled subset (capped dataset)** was used, with the cap size **varying depending on the experiment and model**.

- **Round 1 (Baseline):**  
  A **smaller capped subset** was used to enable faster experimentation and debugging.

- **Later Experiments (Round 2+):**  
  The dataset size was **progressively increased**, allowing the models to benefit from more data and improving generalization.

- **Model Dependency:**  
  The cap size may differ slightly across models (ArcFace, SphereFace, Triplet) based on:
  - Training stability  
  - Convergence speed  

This approach ensures a **fair comparison under controlled conditions** while maintaining **practical training efficiency**.


---

### 🧪 Testing — LFW (Labeled Faces in the Wild)

> **Dataset:** [LFW Facial Recognition — Kaggle](https://www.kaggle.com/datasets/quadeer15sh/lfw-facial-recognition)

The standard benchmark for unconstrained face verification. Models are evaluated on the canonical 6,000 pairs verification protocol.

| Property | Value |
|---|---|
| Subjects | 5,749 identities |
| Images | ~13,233 |
| Pairs (eval) | 1,000 |
| Protocol | Unrestricted, labeled outside data |

---

## Architecture

All three models share the same backbone to ensure a fair comparison. Only the loss function and the final training head differ between implementations.

```
                    Input Image (112×112×3)
                          │
                          ▼
                    Backbone CNN  (e.g., ResNet-50 / MobileNetV2)
                          │
                          ▼
                    Feature Embedding  (512-D L2-normalized vector)
                          │
       ┌──────────────────│─────────────────────┐
       ▼                  ▼                     ▼
  [Triplet Loss]      [ArcFace]           [SphereFace]   
    Head                Head                  Head
```

> **Note:** Backbone choice and embedding dimensionality are kept constant across the first 2 rounds and it has been changed during the third experiment. Differences arise only from the loss formulation.

---

## Experiments

### Round 1 — Initial Training & Baseline Results

> **Objective:** Train all three models on VGGFace2 from scratch (or from the same pretrained backbone checkpoint) and record baseline performance on LFW.

#### ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | *(e.g., ResNet-50)* |
| Embedding Size | *(e.g., 512)* |
| Optimizer | *(e.g., SGD / Adam)* |
| Learning Rate | Triplet Loss: *(0.001, with cosine decay)*<br>ArcFace / SphereFace: *(0.1, with cosine decay)*|
| Batch Size | *(e.g., 128)* |
| Epochs | Triplet Loss: *(62)*<br>Sphere Loss: *(67)*<br>ArcFace: *(72)* |
| ArcFace Margin (m) | *(e.g., 0.5)* |
| ArcFace Scale (s) | *(e.g., 64)* |
| SphereFace Margin (m) | *(e.g., 4)* |
| Triplet Margin | *(e.g., 0.3)* |
| Triplet Mining | *(e.g., Semi-Hard Online Mining)* |
---

#### 📊 Results — Training Performance
| Model |  Training Loss (Final) | Val Loss (Final) |Training Accuracy (Final) | Val Accuracy (Final) |
|---|---|---|---|---|
| **Triplet Loss** | 0.2763 | 0.2866 | 0.1515 | 0.2045 | 
| **SphereFace** | 0.8643 | 7.6491 | 0.9059 | 0.6638 | 
| **ArcFace** | 1.0365 | 9.0069 | 0.9301  | 0.7111  |  

#### 🔍 Results — LFW Verification
| Model | EER ↓ | AUC ↑ | FAR ↓ | FRR ↓ | F1 ↑ 
|---|---|---|---|---|---|
| **Triplet Loss** | 0.1900 | 0.8951 | 0.1900 | 0.1900 | 0.8100 | 
| **SphereFace** | 0.2820 | 0.7893 | 0.2820 | 0.2820 | 0.7180 | 
| **ArcFace** | 0.3230 | 0.7539 | 0.3220 | 0.3240 | 0.6767 | 

---

#### 📉 Training Curves — Round 1

---

## 🔷 Triplet Loss

| Validation Loss                                    | Validation Accuracy                              |
| -------------------------------------------------- | ------------------------------------------------ |
| ![Triplet Val Loss](wandb\ROUND_1\TripletLoss\val_loss.png) | ![Triplet Val Acc](wandb\ROUND_1\TripletLoss\val_acc.png) |

---

## 🔶 SphereFace

| Validation Loss                                          | Validation Accuracy                                    |
| -------------------------------------------------------- | ------------------------------------------------------ |
| ![SphereFace Val Loss](wandb\ROUND_1\SphereFaceLoss\val_loss.png)| ![SphereFace Val Acc](wandb\ROUND_1\SphereFaceLoss\val_loss.png) |

---

## 🔵 ArcFace

| Validation Loss                                    | Validation Accuracy                              |
| -------------------------------------------------- | ------------------------------------------------ |
| ![ArcFace Val Loss](wandb\ROUND_1\ArcFaceLoss\val_loss.png) | ![ArcFace Val Acc](wandb\ROUND_1\ArcFaceLoss\val_loss.png) |

---

## 📌 Note

Additional training curves (e.g., training loss, learning rate schedules, and more detailed metrics) are available in the W&B logs.

📂 You can find the complete logs and visualizations at:

```
wandb/ROUND_1/
```

---

#### 🔍 Round 1 — Insights & Observations

- **ArcFace:** *(Your observations here)*
- **SphereFace:** *(Your observations here)*
- **Triplet Loss:** *(Your observations here)*
- **General:** *(Cross-model comparison observations)*

---
