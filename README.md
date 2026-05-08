# FashionMNIST-Image-Classifier

## Overview 
A small PyTorch image classification project built around the FashionMNIST dataset. The ML pipeline includes data loading, a simple convolutional neural network, training / evaluation / prediction functions, plotting utilities for visualization, and a lightweight test suite, and building / pushing a Docker image.

## What It Does

- Trains a convolutional neural network on FashionMNIST
- Evaluates accuracy on the test set after each epoch
- Saves trained weights to `results/models/model.pt`
- Generates a prediction grid at `results/figures/predictions.png`
- Includes unit tests for config loading, model training and prediction
- Builds and pushes a Docker image to Dockerhub

## Project Layout

```text
.
├── main.py                  # Main training and inference entry point
├── config.yaml              # Runtime configuration
├── scripts/
│   └── plot.py              # Prediction visualization helpers
├── src/imgclf/
│   ├── config.py            # YAML config loader
│   ├── data.py              # FashionMNIST transforms and dataloaders
│   ├── model.py             # SimpleCNN model definition
│   ├── predict.py           # Single-image prediction helper
│   └── train.py             # Training and evaluation functions
└── tests/                   # Pytest suite
```

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- PyYAML

## Create vrtual environment
```bash
python3 -m venv .venv
```

## Activate the environment
```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

The project reads settings from [`config.yaml`]:

```yaml
data:
  batch_size: 64
  num_workers: 2

model:
  num_classes: 10

training:
  num_epochs: 10
  learning_rate: 0.001
  device: cuda

plotting:
  num_images: 12

paths:
  model_save_dir: results/models
  figure_save_dir: results/figures
```

If `cuda` is configured but your local CUDA driver is unavailable or incompatible, switch `training.device` to `cpu`.

## How To Run

Train the model and generate predictions:

```bash
python main.py
```

This will:

1. Download FashionMNIST if needed
2. Train the CNN for the configured number of epochs
3. Save model weights
4. Run a sample prediction
5. Save a prediction figure

## Testing

Run the test suite with:

```bash
pytest
```

Current status at the time of this README update: `5 passed`.

## Notes

- `main.py` is the working entry point for the full pipeline.

