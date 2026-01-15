# CSIRO Image to Biomass Prediction: Experimental Suite

This repository documents a series of ablation studies and optimization experiments for the **CSIRO Image2Biomass** competition. All experiments were conducted in the **Kaggle Notebook environment** utilizing **NVIDIA Tesla P100 or T4 GPUs**.

## Environment & Hardware

* **Platform**: Kaggle Notebooks
* **Accelerator**: NVIDIA P100 / T4 GPU
* **Framework**: PyTorch 2.x, CatBoost, LightGBM, XGBoost
* **Feature Extraction**: DINOv2 (Patch-based embeddings)
* **Training Duration**: 50 Epochs (for CNN-based experiments)

## Experimental Results

### 1. CNN Baseline Summary Metrics
The following table reflects the global averages across early CNN optimization trials.

| Metric | Value | Significance |
| --- | --- | --- |
| **Avg Train Loss** | 0.15376 | Robust training convergence. |
| **Avg Val Loss** | 0.12544 | Efficient error minimization on validation sets. |
| **Best Val $R^2$** | **0.32011** | **Peak Potential:** Highest variance explained in CNN runs. |

---

### 2. DINOv2 + GBDT Experiments (Current State-of-the-Art)
This experiment shifted from raw CNNs to utilizing pre-trained **DINOv2** features. Patch-based embeddings (excluding the CLS token) were averaged and passed to Gradient Boosted Decision Trees (GBDTs). This approach significantly outperformed all previous CNN architectures.

#### 5-Fold Cross-Validation Performance
Results are based on the **Global Weighted $R^2$** metric using competition-specific weights: `Dry_Total_g` (0.5), `GDM_g` (0.2), and 0.1 for the remaining targets.

| Model | Average Weighted $R^2$ | Standard Deviation (±) |
| --- | :---: | :---: |
| **CatBoost** | **0.6289** | **0.0415** |
| **LightGBM** | 0.6181 | 0.0502 |
| **XGBoost** | 0.5539 | 0.0539 |

#### Detailed Fold-by-Fold Results ($R^2$)
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 0.6191 | 0.6285 | 0.6873 | 0.6492 | 0.5604 |
| **LightGBM** | 0.5807 | 0.6072 | 0.7106 | 0.6237 | 0.5682 |
| **XGBoost** | 0.5319 | 0.5610 | 0.6550 | 0.5169 | 0.5046 |

---

### 3. Ablation Studies & Optimization (CNN-based)

#### A. Architecture Depth (`/CNN_5_Conv`)
* **Strategy**: Increasing convolutional blocks from 2 to 5.
* **Finding**: The **3-block** configuration provided the best balance. Models with 4+ blocks showed severe overfitting.

#### B. Augmentation Strategies (`/cnn-augmentation-strat`)
* **Top Performer**: **Medium Augmentation**. "No Augmentation" resulted in negative $R^2$ values, proving that visual variation is mandatory.

#### C. Regularization & Dropout (`/cnn-dropout-regularization`)
* **Top Performer**: **Dropout 0.6 + WD 0.0**. High dropout was the single most effective setting for stabilizing validation $R^2$.

#### D. Kernel Size Study (`/csiro-biomass-cnn-experiment-kernel-size`)
* **Finding**: **3x3 Kernels** achieved the highest individual potential in the CNN suite.

---

### 4. Final Optimized Pipeline Strategy

The production pipeline utilizes the winning DINOv2 + GBDT configuration.

| Component | Optimal Selection | Impact |
| --- | --- | --- |
| **Feature Extractor** | **DINOv2 (Patch Avg)** | Jump in $R^2$ from ~0.32 (CNN) to **~0.63**. |
| **Regressor** | **CatBoost** | Best stability (lowest Std Dev) across all folds. |
| **TTA** | **Horizontal Flips** | Reduces variance in patch-based inference. |
| **Regularization** | **Dropout 0.6** | Essential for preventing negative $R^2$ in deep layers. |

## Project Structure

* **dinov2_gbdt_pipeline/**: Current best performing code (DINOv2 + CatBoost).
* **final_optimized_pipeline/**: Production-ready code and weights for `submission.csv`.
* **CNN_5_Conv/**: Depth analysis proving 3 blocks is the "sweet spot".
* **cnn-augmentation-strat/**: Evidence for mandatory augmentation.
* **cnn-dropout-regularization/**: Establishing the 0.6 dropout baseline.
* **csiro-biomass-cnn-experiment-kernel-size/**: Detailed 3x3 vs 5x5 comparison.

## How to Reproduce

1. **Upload** the notebooks to a Kaggle session.
2. **Enable GPU P100/T4** in the settings.
3. **Ensure** the `csiro-biomass` dataset is attached.
4. **Run** the DINOv2 feature extraction and CatBoost training cells to regenerate results.

## Notebook Links
1. https://www.kaggle.com/code/hamzabinbutt/cnn-dropout-regularization
2. https://www.kaggle.com/code/hamzabinbutt/cnn-augmentation-strat
3. https://www.kaggle.com/code/hamzabinbutt/csiro-biomass-cnn-experiment-kernel-size
4. https://www.kaggle.com/code/hamzabinbutt/cnn-5-conv
5. https://www.kaggle.com/code/hamzabinbutt/base-model-csiro
6. https://www.kaggle.com/code/hamzabinbutt/csiro-biomass-cnn-final-optimized-training-pipel
