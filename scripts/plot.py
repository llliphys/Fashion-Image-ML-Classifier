import matplotlib.pyplot as plt

import torch

from pathlib import Path

from imgclf.model import SimpleCNN
from imgclf.predict import predict_single_image
from imgclf.data import get_transforms
from imgclf.config import load_config

from torchvision import datasets, transforms

import random


def gen_sample_images(num_images: int = 12):
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    dataset = datasets.FashionMNIST(
        root="data",
        train=False,  # test set
        download=True,
        transform=transforms.ToTensor(),
    )

    # Export the specified number of example images
    for i in range(num_images):
        image, label = dataset[i]

        # Convert tensor → PIL image
        image = transforms.ToPILImage()(image)
        image = image.resize((280, 280), resample=0)

        file_path = images_dir / f"img_{i}_label_{label}.png"
        image.save(file_path)

        print(f"Saved: {file_path}")


def plot_predictions(trained_model: SimpleCNN,
                     device: str = "cpu",
                     figure_dir: str = "figures",
                     num_images: int = 12,
                     ):

    
    dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=get_transforms(),
        # transform=transforms.ToTensor(),
    )

    # Grid size
    cols = 4
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(12, 3 * rows))

    indices = random.sample(range(len(dataset)), num_images)

    # for i in range(num_images):
    #     image, true_label = dataset[i]

    for i, idx in enumerate(indices):
        image, true_label = dataset[idx]

        pred_label = predict_single_image(trained_model, image, device)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.axis("off")

        color = "green" if pred_label == true_label else "red"

        plt.title(
            f"True: {true_label} $|$ Predicted: {pred_label}",
            fontsize=15,
            color=color
        )

    plt.savefig(f"{figure_dir}/predictions.png", bbox_inches="tight")
    plt.show()
