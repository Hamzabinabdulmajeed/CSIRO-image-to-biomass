# CSIRO Image to Biomass Prediction

Regression project for estimating pasture biomass.

## Project Description

This repository contains a deep learning baseline for the **CSIRO Image2Biomass** competition. The goal is to predict five specific biomass targets from pasture images using a Convolutional Neural Network (CNN).

## Model Architecture

The architecture is designed to handle regression tasks with variable input sizes:

* **Feature Extractor**: 2D Convolutional layers followed by Batch Normalization for training stability.
* **Pooling**: Adaptive Average Pooling to ensure a fixed-size vector regardless of input image resolution.
* **Output Layer**: A linear regressor with a final ReLU activation to ensure predicted biomass values are never negative.

## Loss and Metrics

The model uses a **Weighted Mean Squared Error (MSE)** loss to align with the competition's evaluation objectives.

### Target Weights

| Target Column | Weight |
| --- | --- |
| Dry_Total_g | 0.5 |
| GDM_g | 0.2 |
| Dry_Clover_g | 0.1 |
| Dry_Grass_g | 0.1 |
| Dry_Weeds_g | 0.1 |

Performance is tracked using the **Global Weighted R-squared (R2)** score across all target categories.

## Installation

1. Clone the repository to your local machine.
2. Install the necessary dependencies:

```bash
pip install -r requirements.txt

```

## Training

The training script is configured to run for **50 epochs**. During the process, it automatically records:

* Training and validation loss.
* Weighted R2 scores for each epoch.
* Periodic model checkpoints.

## Project Structure

* `data/`: Local directory for raw images and CSV metadata (excluded from Git).
* `models/`: Directory for storing trained weights and model exports.
* `src/`: Core Python scripts including the model class, custom loss functions, and the training loop.
* `results/`: Visualizations of training history and error analysis plots.
