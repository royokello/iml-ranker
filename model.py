import torch
import torch.nn as nn
import torch.nn.functional as F

class ComparisonNet(nn.Module):
    def __init__(self):
        super(ComparisonNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, image1, image2):
        # Forward pass for image1
        x1 = F.relu(self.conv1(image1))
        x1 = F.max_pool2d(x1, 2)
        x1 = F.relu(self.conv2(x1))
        x1 = F.max_pool2d(x1, 2)
        x1 = F.relu(self.conv3(x1))
        x1 = F.max_pool2d(x1, 2)
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.fc1(x1))

        # Forward pass for image2
        x2 = F.relu(self.conv1(image2))
        x2 = F.max_pool2d(x2, 2)
        x2 = F.relu(self.conv2(x2))
        x2 = F.max_pool2d(x2, 2)
        x2 = F.relu(self.conv3(x2))
        x2 = F.max_pool2d(x2, 2)
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc1(x2))

        # Combine both representations
        x = torch.abs(x1 - x2)
        x = self.fc2(x)
        return x

