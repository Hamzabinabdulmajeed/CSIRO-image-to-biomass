CSIRO Image to Biomass Prediction
Regression project for estimating pasture biomass.

Project Description
This repository contains a deep learning baseline for the CSIRO Image2Biomass competition. The goal is to predict five biomass targets from pasture images using a Convolutional Neural Network (CNN).

Model Architecture
Feature Extractor: 2D Convolutional layers with Batch Normalization.

Pooling: Adaptive Average Pooling to support various input resolutions.

Output: Linear regressor with ReLU activation to ensure non-negative biomass values.

Loss and Metrics
The model uses a Weighted Mean Squared Error (MSE) loss to match the competition evaluation criteria.

Target Weights:

Dry_Total_g: 0.5

GDM_g: 0.2

Dry_Clover_g: 0.1

Dry_Grass_g: 0.1

Dry_Weeds_g: 0.1

Performance is tracked using a Global Weighted R-squared (R2) score.

Installation
Clone the repository.

Install dependencies:

Bash

pip install -r requirements.txt
Training
The training script runs for 50 epochs. It records the training loss and the weighted R2 score for each epoch.

Project Structure
data/: Local directory for images and CSV files (not synced to GitHub).

models/: Storage for trained model weights.

src/: Python scripts for the model, loss functions, and training loop.

results/: Plots for loss and R2 metric history.