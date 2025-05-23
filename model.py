# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

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

class IMLRankModel(nn.Module):
    """
    Vision Transformer (ViT) based model for image ranking.
    
    Uses Google's Vision Transformer as a backbone for feature extraction
    and implements a pairwise comparison mechanism.
    """

    def __init__(self, num_classes=4, pretrained=True):
        """
        Initializes the IMLRankModel model.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 4.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        """
        super(IMLRankModel, self).__init__()
        
        # Load pre-trained ViT model
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)
            
        # Get the feature dimension from the ViT model
        vit_embedding_dim = self.vit.heads.head.in_features
        
        # Remove the classification head as we'll use our own
        self.vit.heads = nn.Identity()
        
        # Fully connected layer for final classification
        self.fc = nn.Linear(vit_embedding_dim * 2, num_classes)
        
        # Initialize the fully connected layer
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, img1, img2):
        """
        Forward pass of the IMLRankModel.

        Args:
            img1 (torch.Tensor): Tensor of the first image batch (batch_size, 3, H, W).
            img2 (torch.Tensor): Tensor of the second image batch (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Logits indicating the relationship between img1 and img2 (batch_size, num_classes).
        """
        # Process first image
        x1 = self.vit(img1)  # Shape: (batch_size, vit_embedding_dim)
        
        # Process second image
        x2 = self.vit(img2)  # Shape: (batch_size, vit_embedding_dim)
        
        # Concatenate features from both images
        combined = torch.cat((x1, x2), dim=1)  # Shape: (batch_size, vit_embedding_dim * 2)
        
        # Pass through the fully connected layer
        logits = self.fc(combined)  # Shape: (batch_size, num_classes)
        
        return logits
