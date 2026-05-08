import torch

from imgclf.data import get_transforms


def test_transform_output_shape():
    transform = get_transforms()

    image = torch.zeros(28, 28)
    transformed = transform(image.numpy())

    assert transformed.shape == (1, 28, 28)
