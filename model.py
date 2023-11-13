import torch
from torch import flatten, Tensor
from torch.nn import (
    Conv3d,
    Dropout,
    MaxPool3d,
    Linear,
    Module,
    ReLU)


class NaiveCNN(Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()

        hidden_channels = 16

        self.conv1 = Conv3d(in_channels=1, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)

        self.relu = ReLU()
        self.pool = MaxPool3d(kernel_size=2, stride=2)

        self.dropout = Dropout(0.3)

        self.fc1 = Linear(16 * 16 * 2 * hidden_channels, 128)
        self.fc2 = Linear(128, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 5:
            x = x.unsqueeze(1)  # add channel dimension

        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv4(x)))

        x = flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
