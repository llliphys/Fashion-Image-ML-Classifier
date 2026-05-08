from pathlib import Path
import os, sys

import random
import numpy as np

import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from imgclf.config import load_config
from imgclf.data import get_dataloaders
from imgclf.model import SimpleCNN
from imgclf.predict import predict_single_image
from imgclf.train import evaluate, train_epoch
from scripts.plot import plot_predictions


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    """Get the PyTorch device to use for computations."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  


def main():
    try: # load the config file with error handling
        config = load_config("config.yaml")
        random_seed = config.get("random_seed", 23)
        device = config["training"]["device"]
        batch_size = config["data"]["batch_size"]
        num_classes = config["model"]["num_classes"]
        learning_rate = config["training"]["learning_rate"]
        num_epochs = config["training"]["num_epochs"]
        model_dir  = config["paths"]["model_save_dir"]
        figure_dir = config["paths"]["figure_save_dir"]
        num_images = config["plotting"]["num_images"]
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        print("Either the file is missing or contains errors.")
        print("Falling back to default training parameters.")
        random_seed = 23
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 64
        num_classes = 10
        learning_rate = 1e-3
        num_epochs = 20
        model_dir  = "results/models"
        figure_dir = "results/figures"
        num_images = 12

    set_seed(random_seed)
    device = get_device(device)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), f"{model_dir}/model.pt")
    print(f"Saved trained model to {model_dir}/model.pt")

    trained_model = SimpleCNN(num_classes=num_classes)
    trained_model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))

    sample_images, sample_labels = next(iter(test_loader))
    predicted_label = predict_single_image(trained_model, sample_images[0], device)
    true_label = sample_labels[0].item()
    print(f"Single prediction - predicted: {predicted_label}, true: {true_label}")

    plot_predictions(trained_model, device, figure_dir=figure_dir, num_images=num_images)
    print(f"Saved predictions to {figure_dir}/predictions.png")


if __name__ == "__main__":
    main()
