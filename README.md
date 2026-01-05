# CSIRO Image to Biomass Prediction: Experimental Suite

This repository documents a series of ablation studies and optimization experiments for the **CSIRO Image2Biomass** competition. All experiments were conducted in the **Kaggle Notebook environment** utilizing **NVIDIA Tesla P100 or T4 GPUs** for accelerated training.

## Environment & Hardware

* **Platform**: Kaggle Notebooks
* **Accelerator**: NVIDIA P100 / T4 GPU
* **Framework**: PyTorch 2.x
* **Training Duration**: 50 Epochs per experiment

## Experimental Results

### 1. Base Model (`/base_model`)

The initial baseline to establish a performance floor.

* **Strategy**: Simple 3-block CNN without advanced regularization.
* **Key Metrics**:
* **Best Val R²**: 0.2515 (Epoch 47)
* **Best Val Loss**: 0.1007 (Epoch 35)



### 2. Architecture Depth (`/CNN_5_Conv`)

Investigating if deeper networks improve biomass estimation.

* **Strategy**: Increasing convolutional blocks from 2 to 5.
* **Results**: Models with **4+ blocks** showed severe overfitting. **2–3 blocks** were identified as the optimal depth for this dataset size.

### 3. Augmentation Strategies (`/cnn-augmentation-strat`)

Testing the impact of synthetic data variety on generalization.

* **Strategy**: Compared "No Augmentation", "Light", "Medium", and "Strong".
* **Top Performer**: **Medium Augmentation** (Avg Val R²: 0.112). "No Augmentation" resulted in negative R² values, proving that variation is critical for biomass regression.

### 4. Regularization & Dropout (`/cnn-dropout-regularization`)

Fine-tuning the model to handle noise and prevent overfitting.

* **Strategy**: Grid search on Dropout (0.0 to 0.6) and Weight Decay (0.0 to 0.001).
* **Top Performer**: **Dropout 0.6 + WD 0.0001**. High dropout significantly improved validation stability.

### 5. Kernel Size Study (`/csiro-biomass-cnn-experiment-kernel-size`)

Examining spatial receptive fields for pasture imagery.

* **Strategy**: Comparison of 3x3, 5x5, and 7x7 kernels.
* **Finding**: **5x5 Kernels** provided the best balance of local feature extraction and training stability.

### 6. Final Optimized Pipeline (`/csiro-biomass-cnn-final-optimized-training-pipel`)

Consolidation of all winning strategies into a single production pipeline.

* **Config**: 5x5 Kernels, 3 Conv Blocks, Medium Augmentation, and 0.6 Dropout.
* **Best Val R²**: **0.1657** (Aggregated average across targets).

## Project Structure

Each folder contains the specific Kaggle Notebook (`.ipynb`) and the resulting performance plots:

* **base_model/**
* **CNN_5_Conv/**
* **cnn-augmentation-strat/**
* **cnn-dropout-regularization/**
* **csiro-biomass-cnn-experiment-kernel-size/**
* **csiro-biomass-cnn-final-optimized-training-pipel/**

## How to Reproduce

1. **Upload** the notebooks to a Kaggle session.
2. **Enable GPU P100** in the "Accelerator" settings.
3. **Ensure** the competition dataset is attached to the notebook.
4. **Run all cells** to generate the 50-epoch training logs and plots.

## Kaggle Notebooks & Resources

The following notebooks contain the full implementation, training logs, and visualizations for each stage of the project:

* **Base Model**: [View on Kaggle](https://www.kaggle.com/code/hamzabinbutt/base-model-csiro)
* **CNN 5 Conv Layers Experiment**: [View on Kaggle](https://www.kaggle.com/code/hamzabinbutt/cnn-5-conv)
* **Kernel Size Experiment**: [View on Kaggle](https://www.kaggle.com/code/hamzabinbutt/csiro-biomass-cnn-experiment-kernel-size)
* **Augmentation Strategy Study**: [View on Kaggle](https://www.kaggle.com/code/hamzabinbutt/cnn-augmentation-strat)
* **Final Optimized Pipeline**: [View on Kaggle](https://www.kaggle.com/code/hamzabinbutt/csiro-biomass-cnn-final-optimized-training-pipel)
