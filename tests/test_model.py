import torch

from imgclf.model import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)

    y = model(x)

    assert y.shape == (4, 10)
