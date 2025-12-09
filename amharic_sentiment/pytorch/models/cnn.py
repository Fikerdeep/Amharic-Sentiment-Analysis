"""
PyTorch CNN model for Amharic sentiment analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from amharic_sentiment.pytorch.models.base import BaseClassifier


class CNNClassifier(BaseClassifier):
    """
    Convolutional Neural Network for sentiment classification.

    Architecture:
        - Embedding layer
        - Conv1d with ReLU activation
        - Global max pooling
        - Fully connected layers with dropout
        - Sigmoid output

    Example:
        >>> model = CNNClassifier(vocab_size=10000, embedding_dim=100)
        >>> output = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 64,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 1,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize CNN classifier.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_filters: Number of convolutional filters
            kernel_size: Kernel size for convolution
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            num_classes: Number of output classes
            pretrained_embeddings: Pre-trained embeddings
            freeze_embeddings: Whether to freeze embeddings
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings
        )

        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # Convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same'
        )

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Output logits
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Conv1d expects (batch, channels, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Convolution + ReLU
        conv_out = F.relu(self.conv1d(embedded))

        # Global max pooling
        pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)

        # Fully connected layers
        hidden = self.dropout(F.relu(self.fc1(pooled)))
        output = self.fc2(hidden)

        return output.squeeze(-1)
