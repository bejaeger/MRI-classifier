import torch
from torch import Tensor
from torch.nn import (
    Conv3d,
    MaxPool3d,
    Linear,
    Module,
    ReLU)


class SimpleCNN(Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = Linear(128 * 128 * 20 * 32, 128)
        self.fc2 = Linear(128, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 5:
            x = x.unsqueeze(1)  # add channel dimension

        x = self.pool(self.relu(self.conv1(x)))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
