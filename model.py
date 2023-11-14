from torch import flatten, Tensor
from torch import nn
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


class AlexNet3D(Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(AlexNet3D, self).__init__()

        hidden_channels = 48

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, hidden_channels * 2, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2))        
        self.layer3 = nn.Sequential(
            nn.Conv3d(hidden_channels * 4, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(14 * 14 * hidden_channels, 1024),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(1024, num_classes))
        
        self.dropout_conv =  nn.Dropout(0.2)
    
    def forward(self, x):
        if x.ndim < 5:
            x = x.unsqueeze(1)  # add channel dimension
    
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
