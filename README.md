# CSIRO Image to Biomass Prediction: Experimental Suite

This repository documents a series of ablation studies and optimization experiments for the **CSIRO Image2Biomass** competition. All experiments were conducted in the **Kaggle Notebook environment** utilizing **NVIDIA Tesla P100 or T4 GPUs**.

## Environment & Hardware

* **Platform**: Kaggle Notebooks
* **Accelerator**: NVIDIA P100 / T4 GPU
* **Framework**: PyTorch 2.x
* **Training Duration**: 50 Epochs per experiment

## Experimental Results

### 1. Final Summary Metrics (Averaged)

The following table reflects the global averages across all optimization trials and the identifying peak potential.

| Metric | Value | Significance |
| --- | --- | --- |
| **Avg Train Loss** | 0.15376 | Robust training convergence. |
| **Avg Val Loss** | 0.12544 | Efficient error minimization on validation sets. |
| **Avg Train ** | 0.15202 | Stable baseline training performance. |
| **Avg Val ** | 0.03891 | Positive average generalization across all folds. |
| **Best Val ** | **0.32011** | **Peak Potential:** Highest variance explained in a single run. |

---

### 2. Ablation Studies & Optimization

#### A. Architecture Depth (`/CNN_5_Conv`)

Investigating if deeper networks improve biomass estimation.

* **Strategy**: Increasing convolutional blocks from 2 to 5.
* **Finding**: Models with **4+ blocks** showed severe overfitting. The **3-block** configuration provided the best balance, maintaining the lowest validation loss (0.120).

#### B. Augmentation Strategies (`/cnn-augmentation-strat`)

* **Strategy**: Compared "No Aug", "Light", "Medium", and "Strong".
* **Top Performer**: **Medium Augmentation** (Avg Val : 0.112). "No Augmentation" resulted in negative  values, proving that variation is critical for biomass regression.

#### C. Regularization & Dropout (`/cnn-dropout-regularization`)

* **Strategy**: Grid search on Dropout (0.0 to 0.6) and Weight Decay (0.0 to 0.001).
* **Top Performer**: **Dropout 0.6 + WD 0.0**. High dropout was the single most effective setting for maximizing validation .

#### D. Kernel Size Study (`/csiro-biomass-cnn-experiment-kernel-size`)

* **Strategy**: Comparison of 3x3, 5x5, and 7x7 kernels.
* **Finding**: **3x3 Kernels** achieved the highest individual  potential (reaching the **0.320** peak).

---

### 3. Final Optimized Strategy (`/final-optimized-pipeline`)

Consolidation of all winning parameters into the `OptimizedCNN` production class.

| Component | Optimal Selection | Impact |
| --- | --- | --- |
| **Kernel Size** | **3x3** | Maxes out  peak performance. |
| **Blocks** | **3 Conv Blocks** | Captures complex features without overfitting. |
| **Dropout** | **0.6** | Essential for preventing negative  in biomass samples. |
| **Augmentation** | **Light/Medium** | Reduces validation variance. |

## Project Structure

* **base_model/**: established performance floor (Best Val : 0.25).
* **CNN_5_Conv/**: depth analysis proving 3 blocks is the "sweet spot".
* **cnn-augmentation-strat/**: proving augmentation is mandatory for biomass.
* **cnn-dropout-regularization/**: establishing the 0.6 dropout baseline.
* **csiro-biomass-cnn-experiment-kernel-size/**: detailed 3x3 vs 5x5 comparison.
* **final_optimized_pipeline/**: Production-ready code and weights.

## How to Reproduce

1. **Upload** the notebooks to a Kaggle session.
2. **Enable GPU P100** in the settings.
3. **Ensure** the `csiro-biomass` dataset is attached.
4. **Run all cells** to regenerate the logs and plots.
