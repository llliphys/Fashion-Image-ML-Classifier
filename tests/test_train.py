import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from imgclf.model import SimpleCNN
from imgclf.train import train_epoch


def test_train_one_epoch_runs():
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = train_epoch(model, dataloader, criterion, optimizer, device="cpu")

    assert isinstance(loss, float)
    assert loss > 0
