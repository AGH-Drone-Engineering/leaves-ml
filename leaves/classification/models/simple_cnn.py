import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(in_shape[0], 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * in_shape[1] // 4 * in_shape[2] // 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        
        return x
