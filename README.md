---

# CSIRO Image to Biomass Prediction: Experimental Suite

This repository documents a series of ablation studies and optimization experiments for the **CSIRO Image2Biomass** competition. All experiments were conducted in the **Kaggle Notebook environment** utilizing **NVIDIA Tesla P100 or T4 GPUs**.

## Environment & Hardware

* **Platform**: Kaggle Notebooks
* **Accelerator**: NVIDIA P100 / T4 GPU
* **Framework**: PyTorch 2.x, CatBoost, LightGBM, XGBoost
* **Feature Extraction**: DINOv2 (Patch-based embeddings)

## Experimental Results

### 1. Final Summary Metrics (CNN Baselines)

The following table reflects the global averages across early CNN optimization trials.

| Metric | Value | Significance |
| --- | --- | --- |
| **Avg Train Loss** | 0.15376 | Robust training convergence. |
| **Avg Val Loss** | 0.12544 | Efficient error minimization on validation sets. |
| **Best Val ** | **0.32011** | **Peak Potential:** Highest variance explained in CNN runs. |

---

### 2. Ablation Studies & Optimization (CNN)

#### A. Architecture Depth (`/CNN_5_Conv`)

* **Strategy**: Increasing convolutional blocks from 2 to 5.
* **Finding**: Models with **4+ blocks** showed severe overfitting. The **3-block** configuration provided the best balance.

#### B. Augmentation Strategies (`/cnn-augmentation-strat`)

* **Top Performer**: **Medium Augmentation**. "No Augmentation" resulted in negative  values, proving variation is critical.

#### C. Regularization & Dropout (`/cnn-dropout-regularization`)

* **Top Performer**: **Dropout 0.6 + WD 0.0**. High dropout was the most effective setting for maximizing validation .

---

### 3. DINOv2 Patch-Embedding Experiments

This experiment shifted from raw CNNs to utilizing pre-trained **DINOv2** features. Patch-based embeddings (excluding the CLS token) were averaged and passed to Gradient Boosted Decision Trees (GBDTs).

#### 5-Fold Cross-Validation Summary

Results based on the **Global Weighted ** metric (Weights: Dry_Total_g: 0.5, GDM_g: 0.2, Others: 0.1).

| Model | Average Weighted  | Standard Deviation (±) |
| --- | --- | --- |
| **CatBoost** | **0.6289** | **0.0415** |
| **LightGBM** | 0.6181 | 0.0502 |
| **XGBoost** | 0.5539 | 0.0539 |

#### Detailed Fold Results ()

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | --- | --- | --- | --- | --- |
| **CatBoost** | 0.6191 | 0.6285 | 0.6873 | 0.6492 | 0.5604 |
| **LightGBM** | 0.5807 | 0.6072 | 0.7106 | 0.6237 | 0.5682 |
| **XGBoost** | 0.5319 | 0.5610 | 0.6550 | 0.5169 | 0.5046 |

---

### 4. Final Optimized Strategy

The pipeline evolved from custom CNNs to a **DINOv2 + CatBoost** ensemble.

| Component | Optimal Selection | Impact |
| --- | --- | --- |
| **Feature Extractor** | **DINOv2 (Patch Avg)** | Significant jump in  from ~0.32 to ~0.63. |
| **Regressor** | **CatBoost** | Highest stability and best weighted performance. |
| **TTA** | **Horizontal Flips** | Improved inference robustness on patch embeddings. |
| **Regularization** | **Dropout 0.6** | (For CNN variants) Essential for preventing negative . |

## Project Structure

* **base_model/**: Established performance floor (Best Val : 0.25).
* **CNN_5_Conv/**: Depth analysis proving 3 blocks is the "sweet spot".
* **cnn-dropout-regularization/**: Establishing the 0.6 dropout baseline.
* **dinov2-gbdt-pipeline/**: Implementation of DINOv2 features with CatBoost/LGBM/XGBoost.
* **final_optimized_pipeline/**: Production-ready code using CatBoost for final submission.

## How to Reproduce

1. **Upload** the notebooks to a Kaggle session.
2. **Enable GPU (P100 or T4)** in the settings.
3. **Ensure** the `csiro-biomass` dataset is attached.
4. **Run** the DINOv2 feature extraction and CatBoost training cells to generate `submission.csv`.

---

