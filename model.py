# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    BasicBlock for ResNet-like architecture.

    Consists of two convolutional layers with a residual connection.
    """
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initializes the BasicBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer to match dimensions. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Downsampling layer if needed
        self.downsample = downsample
        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the block.
        """
        identity = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out

class CustomResNet(nn.Module):
    """
    Custom ResNet-like architecture tailored for 256x256 images.

    Captures small details effectively and outputs logits for four classes.
    """

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=4):
        """
        Initializes the CustomResNet.

        Args:
            block (nn.Module, optional): Block type to use. Defaults to BasicBlock.
            layers (list, optional): Number of blocks in each of the four layers. Defaults to [2, 2, 2, 2] (ResNet-18).
            num_classes (int, optional): Number of output classes. Defaults to 4.
        """
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer tailored for 256x256 images
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the four layers of the ResNet
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # Output: 64 channels
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Output: 128 channels
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Output: 256 channels
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Output: 512 channels

        # Adaptive average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final fully connected layer for classification
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a layer consisting of multiple blocks.

        Args:
            block (nn.Module): Block type to use.
            out_channels (int): Number of output channels for the blocks.
            blocks (int): Number of blocks to stack.
            stride (int, optional): Stride for the first block. Defaults to 1.

        Returns:
            nn.Sequential: Sequential container of blocks.
        """
        downsample = None
        # Determine if downsampling is needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block with possible downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Initializes the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutional layers with He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch normalization layers
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Initialize fully connected layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, img1, img2):
        """
        Forward pass of the CustomResNet.

        Args:
            img1 (torch.Tensor): Tensor of the first image batch (batch_size, 3, 256, 256).
            img2 (torch.Tensor): Tensor of the second image batch (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: Logits indicating the relationship between img1 and img2 (batch_size, num_classes).
        """
        # Process first image
        x1 = self.conv1(img1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)  # Shape: (batch_size, 512)

        # Process second image
        x2 = self.conv1(img2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)  # Shape: (batch_size, 512)

        # Concatenate features from both images
        combined = torch.cat((x1, x2), dim=1)  # Shape: (batch_size, 1024)

        # Pass through the fully connected layer
        logits = self.fc(combined)  # Shape: (batch_size, num_classes)

        return logits
