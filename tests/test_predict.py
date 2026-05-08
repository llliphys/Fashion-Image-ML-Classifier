import torch

from imgclf.model import SimpleCNN
from imgclf.predict import predict_single_image


def test_predict_single_image_returns_integer_class():
    model = SimpleCNN()
    image = torch.randn(1, 28, 28)

    prediction = predict_single_image(model, image)

    assert isinstance(prediction, int)
    assert 0 <= prediction <= 9
