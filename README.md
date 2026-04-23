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
  - [Round 2 — Investigating Data Scaling for Triplet Loss](#round-2--investigating-data-scaling-for-triplet-loss)
  - [Round 3 — Full Convergence & Fixed Triplet Mining](#round-3--full-convergence--fixed-triplet-mining)
  - [Round 4 — ArcFace Production Training on Scaled Dataset](#round-4--arcface-production-training-on-scaled-dataset)
- [Production System](#production-system)
  - [Database Population — build_database.py](#database-population--build_databasepy)
  - [Real-Time Attendance — realtime_attendance.py](#real-time-attendance--realtime_attendancepy)
- [Installation & Usage](#installation--usage)
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

| # | Paper | Authors | Venue | Link |
|---|---|---|---|---|
| 1 | **FaceNet: A Unified Embedding for Face Recognition and Clustering** *(Triplet Loss)* | Schroff et al. | CVPR 2015 | [📄 arXiv:1503.03832](https://arxiv.org/abs/1503.03832) |
| 2 | **SphereFace: Deep Hypersphere Embedding for Face Recognition** | Liu et al. | CVPR 2017 | [📄 arXiv:1704.08063](https://arxiv.org/abs/1704.08063) |
| 3 | **ArcFace: Additive Angular Margin Loss for Deep Face Recognition** | Deng et al. | CVPR 2019 | [📄 arXiv:1801.07698](https://arxiv.org/abs/1801.07698) |

---

## Datasets

### 🏋️ Training & Validation — VGGFace2 (112×112)

> **Dataset:** [VGGFace2 112×112 — Kaggle](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112)

| Property | Value |
|---|---|
| Subjects | ~9,000 identities |
| Images | ~3.3 million |
| Resolution | 112 × 112 |
| Split Used | Train / Validation |

⚠️ **Important Note on Data Usage**

Due to the large scale of VGGFace2 (~3.3M images), a **controlled capped subset** was used across all experiments, with the cap size varying by round and model based on the specific hypothesis being tested. This approach ensures practical training efficiency while maintaining experimental control.

---

### 🧪 Testing — LFW (Labeled Faces in the Wild)

> **Dataset:** [LFW Facial Recognition — Kaggle](https://www.kaggle.com/datasets/quadeer15sh/lfw-facial-recognition)

| Property | Value |
|---|---|
| Subjects | 5,749 identities |
| Images | ~13,233 |
| Pairs (eval) | 1,000 |
| Protocol | Unrestricted, labeled outside data |

---

## Architecture

All three models share the same backbone to ensure a fair comparison. Only the loss function and training head differ.

```
                    Input Image (112×112×3)
                          │
                          ▼
                    Backbone CNN  (ResNet-50)
                          │
                          ▼
                    Feature Embedding  (512-D L2-normalized vector)
                          │
       ┌──────────────────│─────────────────────┐
       ▼                  ▼                     ▼
  [Triplet Loss]      [ArcFace]           [SphereFace]
    Head                Head                  Head
```

---

## Experiments

### Round 1 — Initial Training & Baseline Results

> **Objective:** Train all three models on a small VGGFace2 subset and record baseline performance on LFW. This round is exploratory — training was intentionally stopped early for ArcFace and SphereFace, and serves as a starting point for hypothesis formation.

#### ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 |
| Embedding Size | 512 |
| Optimizer | SGD |
| Learning Rate | Triplet: 0.001 with cosine decay — ArcFace / SphereFace: 0.1 with cosine decay |
| Batch Size | 128 |
| Epochs | Triplet: 62 — SphereFace: 67 — ArcFace: 72 |
| ArcFace Margin (m) | 0.5 |
| ArcFace Scale (s) | 64 |
| SphereFace Margin (m) | 4 |
| Triplet Margin | 0.3 |
| Triplet Mining | Semi-Hard Online Mining |

---

#### 📊 Results — Training Performance

| Model | Training Loss (Final) | Val Loss (Final) | Training Accuracy (Final) | Val Accuracy (Final) |
|---|---|---|---|---|
| **Triplet Loss** | 0.2763 | 0.2866 | 0.1515 | 0.2045 |
| **SphereFace** | 0.8643 | 7.6491 | 0.9059 | 0.6638 |
| **ArcFace** | 1.0365 | 9.0069 | 0.9301 | 0.7111 |

#### 🔍 Results — LFW Verification

| Model | EER ↓ | AUC ↑ | FAR ↓ | FRR ↓ | F1 ↑ |
|---|---|---|---|---|---|
| **Triplet Loss** | 0.1900 | 0.8951 | 0.1900 | 0.1900 | 0.8100 |
| **SphereFace** | 0.2820 | 0.7893 | 0.2820 | 0.2820 | 0.7180 |
| **ArcFace** | 0.3230 | 0.7539 | 0.3220 | 0.3240 | 0.6767 |

---

#### 📉 Training Curves — Round 1

## 🔷 Triplet Loss

| Validation Loss | Validation Accuracy |
| -------------------------------------------------- | ------------------------------------------------ |
| ![Triplet Val Loss](wandb/ROUND_1/TripletLoss/val_loss.png) | ![Triplet Val Acc](wandb/ROUND_1/TripletLoss/val_acc.png) |

## 🔶 SphereFace

| Validation Loss | Validation Accuracy |
| -------------------------------------------------------- | ------------------------------------------------------ |
| ![SphereFace Val Loss](wandb/ROUND_1/SphereFaceLoss/val_loss.png) | ![SphereFace Val Acc](wandb/ROUND_1/SphereFaceLoss/val_acc.png) |

## 🔵 ArcFace

| Validation Loss | Validation Accuracy |
| -------------------------------------------------- | ------------------------------------------------ |
| ![ArcFace Val Loss](wandb/ROUND_1/ArcFaceLoss/val_loss.png) | ![ArcFace Val Acc](wandb/ROUND_1/ArcFaceLoss/val_acc.png) |

📂 Full W&B logs: `wandb/ROUND_1/`

---

#### 🔍 Round 1 — Insights & Observations

- **ArcFace & SphereFace:** Training was intentionally stopped before full convergence (ArcFace at epoch 72, SphereFace at epoch 67 out of a planned 100). Both models showed healthy learning curves — training accuracy reached ~93% and ~90% respectively, with validation accuracy at ~71% and ~66%. These results are expected to improve significantly once trained to completion in later rounds.

- **Triplet Loss:** Presents a contradictory and still unexplained result. Training performance was very poor — the loss barely moved across 63 epochs (0.297 → 0.287) and validation accuracy only reached ~20%, far below ArcFace and SphereFace. Yet it achieved the **best LFW verification results** of all three models (EER=0.190, AUC=0.895, F1=0.810). This contradiction is not yet fully understood.

- **Open Question — Why does Triplet verify best despite training worst?** Two hypotheses were formed:
  - **Data hypothesis:** The small dataset may be limiting ArcFace and SphereFace's ability to generalize to unseen LFW identities, while metric learning is inherently better suited to open-set verification even with limited data.
  - **Mining hypothesis:** Semi-hard mining may not be generating sufficiently informative triplets, causing the flat loss curve. The model may be learning a rough but usable embedding space early on, then stagnating.

- **General:** No definitive conclusions from Round 1 alone. The priority going into Round 2 was to test the data hypothesis for Triplet specifically, before committing resources to scaling all three models.

---

### Round 2 — Investigating Data Scaling for Triplet Loss

> **Objective:** Test whether the small dataset size was the root cause of Triplet Loss's poor training performance. ArcFace and SphereFace were not scaled in this round — they were already converging well and scaling them would consume significant compute without answering the core question about Triplet first.

#### 🔬 What Changed from Round 1

| Change | Round 1 | Round 2 |
|---|---|---|
| Triplet training data | Small capped subset | **Larger capped subset** |
| ArcFace / SphereFace data | Small capped subset | Unchanged — not retrained this round |
| Backbone | ResNet-50 | ResNet-50 *(unchanged)* |
| Everything else | — | Identical to Round 1 |

> **Rationale:** Scaling all three models at once would consume significant compute resources without a clear hypothesis to test for ArcFace and SphereFace. The experiment was scoped to Triplet only — the model with the unexplained behavior — to keep it focused and resource-efficient.

---

#### ⚙️ Training Configuration — Round 2 (Triplet only)

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 |
| Embedding Size | 512 |
| Optimizer | SGD |
| Learning Rate | 0.001 with cosine decay |
| Batch Size | 128 |
| Data Cap (identities) | 1000 |
| Triplet Margin | 0.3 |
| Triplet Mining | Semi-Hard Online Mining |

---

#### ⚠️ Important Note on Training Duration

Round 2 used a larger dataset, so fewer epochs were completed within the same compute budget. A direct epoch-for-epoch comparison with Round 1 is not valid — convergence with more data naturally takes longer in wall-clock time. Results should be interpreted through convergence trends, not epoch counts.

---

#### 🔍 Key Finding — Data Was Not the Problem

Triplet Loss exhibited the **same flat loss curve and low training accuracy** in Round 2 as in Round 1, despite the larger dataset. The loss trajectory and validation accuracy were nearly identical to Round 1 at the same training stage.

**This rules out data size as the primary cause of Triplet's poor training performance.**

**Conclusion:** The bottleneck is the **mining strategy**. Semi-hard online mining is not generating sufficiently informative triplets — most triplets are already satisfied before the optimizer step, the loss hovers near zero, and the model stops learning. This also reframes the Round 1 LFW result — the useful embedding was likely learned in the early epochs before mining collapsed, and does not reflect the model's true potential.

Triplet Loss training was stopped in this round. **Fixing the mining strategy is a prerequisite before any further Triplet experiments.** This will be addressed in Round 3.

---

#### 📉 Training Curves — Round 2 (Triplet)

## 🔷 Triplet Loss

| Validation Loss | Validation Accuracy |
| -------------------------------------------------- | ------------------------------------------------ |
| ![Triplet Val Loss](wandb/ROUND_2/TripletLoss/val_loss.png) | ![Triplet Val Acc](wandb/ROUND_2/TripletLoss/val_acc.png) |

📂 Full W&B logs: `wandb/ROUND_2/`

---

#### 🔍 Round 2 — Insights & Observations

- **Triplet Loss:** The data hypothesis has been falsified. Scaling the dataset produced no meaningful change in training behavior — the loss curve remained flat and training accuracy stayed low. The mining strategy is confirmed as the root cause. This is a valuable negative result: it prevents wasting further compute on data scaling before the underlying issue is fixed.

- **ArcFace & SphereFace:** Not retrained in this round. Resources were deliberately preserved for Round 3, where all three models will be brought to full convergence under controlled conditions.

---

### Round 3 — Full Convergence & Fixed Triplet Mining 

> **Objective:** Address all outstanding issues identified across Rounds 1 and 2
> in a single controlled experiment. ArcFace and SphereFace are trained to full
> convergence (100 epochs) on the same small dataset as Round 1. Triplet Loss is
> retrained from scratch with a corrected batch construction strategy and a more
> aggressive mining approach. This round will produce the first clean,
> fully-converged comparison across all three models under fair conditions.

#### 🔬 What Will Change

| Change | Previous Rounds | Round 3 |
|---|---|---|
| ArcFace / SphereFace epochs | Stopped early (~67–72) | **Full Training — 200 epochs for arcface/sphereface and 100 epochs for triplet** |
| Triplet batch construction | Random shuffle (broken) | **PK Sampling — P identities × K images per batch** |
| Triplet mining strategy | Semi-hard (collapsed due to empty positives) | **BatchHard — hardest positive + hardest negative per anchor** |
| Dataset size | Small (R1) / Larger for Triplet (R2) | **Small dataset — same as Round 1 for all models** |
| Backbone | ResNet-50 | ResNet-50 *(unchanged)* |

> **Rationale:** Root cause analysis of Round 2 revealed that the Triplet mining
> failure was not caused by the loss implementation or the dataset size, but by
> the batch construction strategy. Random shuffling produced batches where most
> anchors had zero valid positives, silently zeroing out the loss from the early
> epochs onward. Fixing this requires two coordinated changes: a custom PK
> sampler that guarantees multiple images per identity per batch, and switching
> to BatchHard mining which is robust once positives are guaranteed. ArcFace and
> SphereFace require no structural changes — only full convergence. Scaling data
> remains deferred until all three models are confirmed to be learning correctly
> under fair conditions.

#### 🔍 Root Cause — Why Triplet Mining Was Silently Broken

The mining failure had nothing to do with the loss function code itself.
The code was correct. The problem was what it received as input.

**The batch construction problem:** The standard `DataLoader` with random
shuffling filled each batch of 512 images by sampling randomly across all
~9,000 identities. With that many identities, most batches contained only
one image per person. This means for most anchors in the batch, there was
no second image of the same identity — zero valid positives.

**How this silently zeroed the loss:** When an anchor has no valid positive
in the batch, the positive mask is all-False. The distance to the
"hardest positive" is computed as 0.0 by default. Semi-hard mining then
searches for negatives in the range `(0.0, 0.0 + margin)`. With a
pretrained backbone, almost all pairwise distances already exceed the
margin, so no semi-hard negatives are found either. The fallback to hard
mining still computes loss against `d_ap = 0.0`, which also collapses to
near zero. **The model received near-zero gradients from the very first
epoch and never recovered.**

**Why the Round 1 LFW result was misleading:** The pretrained ResNet-50
backbone already produced usable embeddings before any triplet training
began. The small amount of learning that occurred in the rare early batches
where two images of the same person happened to co-occur was enough to
slightly refine the backbone. The LFW result reflects the pretrained
backbone doing most of the work, not genuine triplet learning.

#### 📊 Results — Training Performance

| Model | Training Loss (Final) | Val Loss (Final) | Training Accuracy (Final) | Val Accuracy (Final) |
|---|---|---|---|---|
| **Triplet Loss** | *0.5191* | *0.1576* | *0.5999* | *0.3912* |
| **SphereFace** | *0.0155* | *4.7343* | *0.9988* | *0.8278* |
| **ArcFace** | *0.0301* | *5.7666* | *0.9984* | *0.8301* |

#### 🔍 Results — LFW Verification

| Model | EER ↓ | AUC ↑ | FAR ↓ | FRR ↓ | F1 ↑ |
|---|---|---|---|---|---|
| **Triplet Loss** | *0.2740* | *0.7974* | *0.2740* | *0.2740* | *0.7260* |
| **SphereFace** | *0.2340* | *0.8464* | *0.2340* | *0.2340* | *0.7660* |
| **ArcFace** | *0.2740* | *0.7808* | *0.2740* | *0.2740* | *0.7260* |

#### 📈 Round 1 vs Round 3 — Improvement Summary

| Model | EER R1 | EER R3 | ΔEER | AUC R1 | AUC R3 | ΔAUC |
|---|---|---|---|---|---|---|
| **Triplet Loss** | 0.1900 | 0.2740 | +0.084 ↑ worse | 0.8951 | 0.7974 | −0.098 ↓ |
| **SphereFace**   | 0.2820 | 0.2340 | −0.048 ↓ better | 0.7893 | 0.8464 | +0.057 ↑ |
| **ArcFace**      | 0.3230 | 0.2740 | −0.049 ↓ better | 0.7539 | 0.7808 | +0.027 ↑ |
---

#### 📉 Training Curves — Round 3

## 🔷 Triplet Loss

| Validation Loss | Validation Accuracy |
| -------------------------------------------------- | ------------------------------------------------ |
| ![Triplet Val Loss](wandb/ROUND_3/TripletLoss/val_loss.png) | ![Triplet Val Acc](wandb/ROUND_3/TripletLoss/val_acc.png) |

## 🔶 SphereFace

| Validation Loss | Validation Accuracy |
| -------------------------------------------------------- | ------------------------------------------------------ |
| ![SphereFace Val Loss](wandb/ROUND_3/SphereFaceLoss/val_loss.png) | ![SphereFace Val Acc](wandb/ROUND_3/SphereFaceLoss/val_acc.png) |

## 🔵 ArcFace

| Validation Loss | Validation Accuracy |
| -------------------------------------------------- | ------------------------------------------------ |
| ![ArcFace Val Loss](wandb/ROUND_3/ArcFaceLoss/val_loss.png) | ![ArcFace Val Acc](wandb/ROUND_3/ArcFaceLoss/val_acc.png) |

📂 Full W&B logs: `wandb/ROUND_3/`

---

#### 🔍 Round 3 — Insights & Observations

**ArcFace** improved from EER 0.323 → 0.274, confirming that the Round 1
result was degraded by early stopping. Training accuracy reached 99.8% with
val accuracy at 83.0%, indicating the model converged strongly on the
training identities. The gap between train and val loss (0.030 vs 5.767)
reflects expected overfitting on a 1000-identity subset — the model has
memorized training identities well but generalizes modestly to unseen LFW
pairs. This is a dataset size constraint, not a model failure.

**SphereFace** was the strongest performer in Round 3 with EER 0.234 and
AUC 0.846 — the best LFW result of any model in any round except Triplet's
Round 1 result (which has since been reattributed to the pretrained backbone).
Train accuracy reached 99.9% with val accuracy at 82.8%, nearly matching
ArcFace. The slightly better LFW generalization over ArcFace suggests that
the multiplicative angular margin may produce more separable embeddings on
this particular dataset size, though the difference (0.234 vs 0.274 EER)
is within the margin of a single rerun and should not be over-interpreted.

**Triplet Loss** produced the most instructive result of Round 3. Despite
the PK sampler fix guaranteeing valid positives per batch and switching to
BatchHard mining, LFW performance degraded compared to Round 1 (EER 0.274
vs 0.190). Training loss only reached 0.519 — far above what a well-trained
triplet model should achieve — and val accuracy of 39.1% remains far below
the margin-based models. Two explanations are most likely:

- The Round 1 result (EER 0.190) was genuinely produced by the pretrained
  ResNet-50 backbone with minimal learned refinement, not by triplet training.
  Round 3 confirms this: a properly trained triplet model on this dataset
  performs worse than the backbone alone, because the gradients from
  BatchHard mining on 1000 identities are moving the embedding space away
  from the pretrained initialization rather than improving it.

- BatchHard mining on a small identity set may be too aggressive — the
  hardest negatives across only 1000 identities are not hard enough to
  force useful metric learning, causing the model to chase near-identical
  embeddings rather than building discriminative structure.

**Overall conclusion for Round 3:** SphereFace edges ArcFace on LFW
generalization, both substantially outperform Triplet Loss under fair
training conditions, and Triplet Loss's Round 1 result has been explained
as a pretrained backbone artifact rather than genuine metric learning.
The margin-based approaches are the correct choice for this system.

## Final Model Selection

Based on results across all three rounds under controlled, identical
conditions, **SphereFace is selected as the primary model** for the
attendance system, with ArcFace retained as a close alternative.

| Criterion | Triplet | ArcFace | SphereFace |
|---|---|---|---|
| Best LFW EER | 0.274 (R3) | 0.274 (R3) | **0.234 (R3)** |
| Best AUC | 0.797 (R3) | 0.781 (R3) | **0.846 (R3)** |
| Training stability | Poor | Good | Good |
| Convergence speed | Slow | Fast | Moderate |
| Sensitivity to batch construction | Very high | None | None |
| Recommended for deployment | No | Yes (fallback) | **Yes (primary)** |

**Why not Triplet Loss:** Triplet's Round 1 result (EER 0.190) was a
pretrained backbone artifact. Under fair training conditions in Round 3,
it is the weakest of the three models and the most sensitive to
engineering decisions (batch construction, mining strategy, batch size).
It is not suitable for a production system without significantly more
engineering effort and a much larger dataset.

**Why SphereFace over ArcFace:** SphereFace achieved lower EER (0.234 vs
0.274) and higher AUC (0.846 vs 0.781) on LFW in Round 3 under identical
training conditions. While ArcFace is generally considered the stronger
method at scale (and achieves better results on large datasets in the
literature), on this constrained 1000-identity setup SphereFace's
multiplicative margin appears to produce slightly more generalizable
embeddings. ArcFace is retained as the deployment fallback given its
stronger theoretical guarantees and wider community validation.

<!-- normally val loss ≥ train loss. This suggests either your val set overlaps with train identities, or the PK sampler is only applied to train and val batches are constructed differently. -->
---

## 🤗 Trained Models (Hugging Face)

| Model | Link |
|---|---|
| Triplet Loss | https://huggingface.co/AbdoSaad24/TripletLossModels |
| ArcFace | https://huggingface.co/AbdoSaad24/BestArcFaceModel |
| SphereFace | https://huggingface.co/AbdoSaad24/BestSphereFaceModel |


---

### Round 4 — ArcFace Production Training on Scaled Dataset

> **Objective:** Scale ArcFace training beyond the 1000-identity research constraint toward a production-grade configuration, using a significantly larger data budget to assess how much generalization headroom remains before deployment.

#### 🔬 Why ArcFace Was Chosen Over SphereFace for Production

While SphereFace achieved a marginally better EER in Round 3 (0.234 vs 0.274), **ArcFace was selected as the production model** for the following reasons:

- **Geometric interpretability:** ArcFace's additive angular margin has a direct geometric interpretation — it enforces a fixed angular decision boundary on the hypersphere, making the decision threshold directly tunable at inference time. SphereFace's multiplicative margin distorts the angular space non-uniformly, making threshold calibration less predictable across unseen identities.

- **Numerical stability at scale:** The multiplicative margin in SphereFace requires a piecewise cosine formulation that becomes numerically sensitive as the number of classes increases. ArcFace's additive formulation remains stable regardless of class count, making it significantly more suitable for scaling to thousands of identities in a production gallery.

- **Round 3 margin is within noise:** The 0.040 EER gap between SphereFace and ArcFace in Round 3 is well within the variance of a single evaluation run on 1000 LFW pairs. It is not a statistically reliable signal to override all of the above engineering considerations.


#### ⚙️ Training Configuration — Round 4 (ArcFace only)

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet-50 |
| Embedding Size | 512 |
| Optimizer | SGD |
| Learning Rate | 0.1 with cosine decay |
| Batch Size | 512 |
| Identities | **3,000** (vs 1,000 in Round 3) |
| Images per Identity | **~200** (vs ~100 in Round 3) |
| Epochs | **100** |
| ArcFace Margin (m) | 0.5 |
| ArcFace Scale (s) | 64 |

> This represents a ~6× increase in total training images over Round 3, while remaining a controlled subset of the full VGGFace2 dataset. The goal was to confirm that ArcFace's generalization scales predictably with data, not to achieve final production accuracy.

#### 📊 Results — Training Performance

| Metric | Value |
|---|---|
| Training Accuracy (Final) | **0.99** |
| Validation Accuracy (Final) | **0.85** |
| Training Loss (Final) | **0.03** |
| Validation Loss (Final) | **4.00** |

The improvement in validation accuracy from 83.0% (Round 3, 1000 identities) to **85.0%** (Round 4, 3000 identities) confirms that ArcFace's generalization scales consistently with data — even a 3× increase in identity count produces a measurable gain. The train/val loss gap remains, which is expected: the model is trained on a closed set of 3000 identities and evaluated on an open-set LFW protocol. This gap will narrow further as the training set grows.

> 🤗 **Trained model available on Hugging Face:** [AbdoSaad24/IndustrialFaceRecognition](https://huggingface.co/AbdoSaad24/IndustrialFaceRecognition)

#### 🔍 Round 4 — Insights & Observations

The Round 4 results are encouraging and follow the expected scaling behavior of ArcFace. Validation accuracy improved over Round 3 despite the open-set nature of LFW, and training converged cleanly to near-zero loss. The remaining validation gap is attributable entirely to dataset size — with more identities covering a wider distribution of poses, lighting, and demographics, the embedding space becomes more generalizable to unseen faces.

**Scaling projection:** The trend across rounds is clear — validation accuracy improves monotonically as training data grows. Extrapolating this trend, training on the full VGGFace2 dataset (~9,000 identities, ~3.3M images) would be expected to push validation accuracy toward 90%+ and bring LFW EER into the 0.05–0.10 range, consistent with published ArcFace results at full scale.

---


## Production System

With ArcFace selected and a production-grade checkpoint trained, the system was packaged into two standalone scripts that form the full attendance pipeline.

### Database Population — `build_database.py`

> 📓 **Full walkthrough notebook:** [Kaggle — FaceRecognition ArcFace](https://www.kaggle.com/code/abdelrhmansaadidrees/2-facerecognition-arcface)

This script is the **one-time setup step** that registers known identities into the system before any live recognition can happen. Given a folder of identity images (a subset of VGGFace2), it does the following:

For each person, it splits their images into two groups — a **gallery set** (10 images) that gets stored in the database, and a **probe set** (5 images) that is held out for evaluation and never seen by the database. Each gallery image is passed through the trained ArcFace backbone to produce a 512-dimensional embedding, which is stored in **ChromaDB** (a vector database optimized for similarity search) alongside metadata linking it to the person's record in **MongoDB**.

After all identities are registered, the script immediately runs a **Top-K accuracy evaluation**: each held-out probe image is embedded and queried against ChromaDB to check whether the correct person appears in the top-1, top-3, or top-5 nearest neighbours. This gives a direct, real-world accuracy estimate of how the system will perform at inference time.

#### 📊 Database Evaluation Results (50 identities, 250 probes)

| Metric | Value |
|---|---|
| Total probe images | 250 |
| Failed reads | 0 |
| **Top-1 Accuracy** | **1836 / 2000  (91.80%)** |
| **Top-3 Accuracy** | **1914 / 2000  (95.70%)** |
| **Top-5 Accuracy** | **1937 / 2000  (96.85%)** |

A **92% Top-1 accuracy** means the system correctly identifies the person on the first try 9 times out of 10 — without ever having seen those probe images during training or registration. The Top-3 and Top-5 numbers (96–97%) confirm that even when the top result is wrong, the correct identity is almost always retrieved within the first few candidates, which is useful for any system that presents a shortlist for human confirmation.

> Note: this evaluation was run on a sample of 50 identities drawn from the VGGFace2 dataset. Performance on a larger, more diverse gallery is expected to evolve as the training dataset grows.

---

### Real-Time Attendance — `realtime_attendance.py`

This script is the **live inference component** — it connects a webcam to the registered database and runs face recognition in real time, frame by frame.

In plain terms: the webcam feed is read continuously. For every frame, MTCNN detects any faces present and crops them out. Each crop is preprocessed (resized, normalized) exactly as during training, then passed through the ArcFace backbone to produce a 512-D embedding. That embedding is immediately queried against ChromaDB to find the closest registered face. If the similarity score exceeds a configurable threshold, the person's name is retrieved from MongoDB and drawn on screen alongside their similarity score. If no match is found above the threshold, the face is labelled "Unknown". A cooldown timer prevents the same person from being logged to attendance multiple times within a short window.

The result is a live annotated video feed showing detected faces, their identified names, similarity scores, and a running FPS counter — the complete attendance system running end-to-end.

```
Webcam Frame
    │
    ▼
MTCNN Face Detection
    │  (one crop per detected face)
    ▼
ArcFace Backbone  →  512-D Embedding
    │
    ▼
ChromaDB Nearest-Neighbour Query
    │
    ├─ similarity ≥ threshold  →  MongoDB name lookup  →  Draw name + score (green box)
    └─ similarity < threshold  →  "Unknown"  (red box)
    │
    ▼
Annotated Live Frame  +  Attendance Log (stdout, with per-person cooldown)
```

---

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/AbdelrahmanSaadIdress/Face-Recognition-System.git
cd face-recognition-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Populate the database with known identities
python build_database.py \
    --checkpoint checkpoints/arcface_resnet50_production.pt \
    --dataset    data/raw/Identities \
    --config     configs/base.yaml

# 4. Run real-time attendance
python realtime_attendance.py \
    --checkpoint checkpoints/arcface_resnet50_production.pt \
    --config     configs/base.yaml

# Use a video file instead of webcam
python realtime_attendance.py \
    --checkpoint checkpoints/arcface_resnet50_production.pt \
    --source path/to/video.mp4
```

---

## 🙏 Acknowledgements
This project was built as a self-directed research effort to understand the practical tradeoffs between modern face recognition loss functions — not just theoretically, but through real training runs, real failures, and real debugging. The four rounds of controlled experiments taught more about metric learning, batch construction, and embedding geometry.


>> **Built with PyTorch · Trained on VGGFace2 · Evaluated on LFW
If you found this project useful, consider starring the repository or checking out the trained models on Hugging Face.**