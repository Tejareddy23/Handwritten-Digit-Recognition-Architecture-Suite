"""
Deep Learning Model Architectures.

This module provides two fundamental Deep Learning architectures:
1. MultiLayerPerceptron (MLP) - Dense/Feedforward Neural Network
2. ConvolutionalNeuralNetwork (CNN) - Convolutional feature extraction network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron (MLP / Deep Feedforward Network).

    DL Algorithms / Mechanisms involved:
    - Linear Transformations: z = Wx + b
    - Non-linear Activation: ReLU(z) = max(0, z)
    - Batch Normalization (1D): Stabilizes distribution of hidden activations across mini-batches
    - Dropout: Regularization technique zeroing random activations to prevent co-adaptation/overfitting
    - Output Logits: Linear projection to class dimension (evaluated with Cross-Entropy Loss)
    """

    def __init__(self, input_dim=28 * 28, hidden_dim1=256, hidden_dim2=128, num_classes=10, dropout_rate=0.2):
        super(MultiLayerPerceptron, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3 (Output logits)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        # Flatten [Batch, 1, 28, 28] -> [Batch, 784]
        x = self.flatten(x)
        
        # Block 1: Linear -> BN -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Block 2: Linear -> BN -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output Logits
        out = self.fc3(x)
        return out


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Convolutional Neural Network (CNN).

    DL Algorithms / Mechanisms involved:
    - 2D Convolution: Discrete spatial cross-correlation with learnable filter kernels
    - Spatial Invariance & Parameter Sharing: Detects local visual features (edges, curves, corners)
    - Max Pooling: Non-linear downsampling that provides translation tolerance and reduces dimensionality
    - Batch Normalization (2D & 1D): Accelerates convergence and acts as mild regularizer
    - Fully Connected Classifier: High-level reasoning mapping extracted visual features to class scores
    """

    def __init__(self, in_channels=1, num_classes=10, dropout_rate=0.25):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        # Convolutional Block 1: [B, 1, 28, 28] -> Conv -> [B, 16, 28, 28] -> Pool -> [B, 16, 14, 14]
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2: [B, 16, 14, 14] -> Conv -> [B, 32, 14, 14] -> Pool -> [B, 32, 7, 7]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_conv = nn.Dropout2d(dropout_rate)
        
        # Fully Connected Classification Head: 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop_conv(x)
        
        # Flatten features: [B, 32, 7, 7] -> [B, 1568]
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        out = self.fc2(x)
        return out


def build_model(model_name="cnn"):
    """
    Factory function to instantiate the requested model architecture.
    """
    model_name = model_name.lower().strip()
    if model_name == "cnn":
        return ConvolutionalNeuralNetwork()
    elif model_name == "mlp":
        return MultiLayerPerceptron()
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose between 'cnn' and 'mlp'.")


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
