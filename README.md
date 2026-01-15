**CSIRO Image2Biomass**
## 1. Phase 1: Custom CNN Baseline & Optimization

The project began by attempting to train specialized CNNs from scratch to establish a baseline for biomass estimation.

### Core Experiments

* **Architecture Depth (`/CNN_5_Conv`)**: Models were tested with 2 to 5 convolutional blocks. We discovered a "sweet spot" at **3 blocks**; deeper networks (4+) led to severe overfitting and validation loss divergence.
* **Kernel Size Study**: A comparison between 3x3 and 5x5 kernels showed that **3x3 kernels** provided better spatial resolution for fine-grained biomass textures.
* **Regularization Strategies**: This was the most critical factor for CNNs. High **Dropout (0.6)** was necessary to stabilize the validation , as lower values often resulted in negative scores.
* **Augmentation Strategy**: "Medium Augmentation" (flips, rotations, brightness) was found to be mandatory. Without it, the models failed to generalize to the varied lighting and growth conditions of the Australian pastures.

### Phase 1 Limitations

Despite optimization, the best CNN achieved a peak ** of ~0.32**. The small dataset size made it difficult for the network to learn robust features from scratch.

---

## 2. Phase 2: DINOv2 + GBDT (Current State-of-the-Art)

To achieve a breakthrough, we pivoted to a Transfer Learning approach using Meta’s **DINOv2**, a self-supervised Vision Transformer.

### The New Strategy

Instead of training a backbone, we used a **frozen DINOv2** to extract rich, high-dimensional features.

1. **Patch-Based Embeddings**: Unlike standard models that use a single global token (CLS), we extracted patch-level embeddings to capture local biomass density.
2. **Global Aggregation**: These patch features were averaged to create a robust image descriptor.
3. **GBDT Head**: The embeddings were fed into Gradient Boosted Decision Trees (CatBoost, LightGBM, and XGBoost).

### 5-Fold Performance Jump

The shift to DINOv2 nearly **doubled** the predictive power of our pipeline.

| Model | Avg Weighted  | Key Strength |
| --- | --- | --- |
| **CatBoost** | **0.6289** | Most stable across folds; lowest variance. |
| **LightGBM** | 0.6181 | Fastest training; competitive accuracy. |
| **XGBoost** | 0.5539 | Solid baseline but struggled with embedding noise. |

---

## 3. Final Optimized Pipeline

The production-ready pipeline combines the best of both phases.

* **Feature Extractor**: DINOv2 (Patch Averaging) for superior spatial understanding.
* **Regressor**: **CatBoost** using a weighted objective to match the competition metric.
* **Inference**: Includes **Test-Time Augmentation (TTA)** with horizontal flips to further reduce prediction variance.
* **Metric Weighting**: The final model is optimized for the weighted  formula:
* `Dry_Total_g` (0.5 weight)
* `GDM_g` (0.2 weight)
* All others (0.1 weight)



---

## 4. Notebook Links Summary

### CNN Ablation Suite

* [CNN Dropout & Regularization](https://www.kaggle.com/code/hamzabinbutt/cnn-dropout-regularization)
* [Augmentation Strategy Analysis](https://www.kaggle.com/code/hamzabinbutt/cnn-augmentation-strat)
* [Kernel Size Study](https://www.kaggle.com/code/hamzabinbutt/csiro-biomass-cnn-experiment-kernel-size)
* [CNN 5-Conv Depth Analysis](https://www.kaggle.com/code/hamzabinbutt/cnn-5-conv)
* [Base Model CSIRO](https://www.kaggle.com/code/hamzabinbutt/base-model-csiro)

### DINOv2 & Final Results

* [DINOv2 + GBDT Experimental](https://www.kaggle.com/code/hamzabinbutt/dinvov2-gb)
* [Final Optimized Training Pipeline](https://www.kaggle.com/code/hamzabinbutt/csiro-biomass-cnn-final-optimized-training-pipel)
