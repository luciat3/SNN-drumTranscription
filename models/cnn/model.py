# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

"""
Class nn.Module from pyTorch is the base class for all neural network modules.
Changes behaviour depending of train or evaluation, moves to device GPU/CPU
etc.
"""
class DrumCNN(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.3):
        """
        :param num_classes: (14)
        :param dropout: probability of randomly zeroing the input
        """
        # initializes nn.Module
        super().__init__()

        # pipeline
        self.features = nn.Sequential(
            # Block 1
            # 2D convolution (1 channel (espectrogram is 1), 16 filters (baseline), 3x3 (detects patterns without high cost))
            # 16 filters - 16 versions of the spectrogram 
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # Batch normalization in 16 channels 
            nn.BatchNorm2d(16),
            # Mantains positive values, turns negative to 0
            nn.ReLU(inplace=True),
            # Reduces dimension by half
            nn.MaxPool2d(kernel_size=(2, 2)),  # (80,86) -> (40,43)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (40,43) -> (20,21)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (20,21) -> (10,10)

            # Reduces overfitting by randomly turning off some elements in training
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Sequential(
            # Flattens batch from [B, 64, 10, 10] to [B, 6400]
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [B, 1, 80, 86]
        x = self.features(x)
        return self.classifier(x)
