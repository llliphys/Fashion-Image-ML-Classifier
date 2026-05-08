import torch

from imgclf.model import SimpleCNN


def predict_single_image(
    model: SimpleCNN, image: torch.Tensor, device: str = "cpu"
) -> int:
    model.eval()
    model.to(device)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    return prediction
