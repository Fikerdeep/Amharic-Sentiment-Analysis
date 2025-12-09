"""
PyTorch CNN-BiLSTM hybrid model for Amharic sentiment analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from amharic_sentiment.pytorch.models.base import BaseClassifier


class CNNBiLSTMClassifier(BaseClassifier):
    """
    Hybrid CNN-BiLSTM for sentiment classification.

    Combines CNN for local feature extraction with BiLSTM for
    capturing long-range dependencies.

    Architecture:
        - Embedding layer
        - Conv1d with ReLU
        - Max pooling
        - Bidirectional LSTM
        - Fully connected output
        - Sigmoid output

    Example:
        >>> model = CNNBiLSTMClassifier(vocab_size=10000, embedding_dim=32)
        >>> output = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        num_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 4,
        lstm_hidden_dim: int = 64,
        dropout_conv: float = 0.2,
        dropout_lstm: float = 0.3,
        num_classes: int = 1,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize CNN-BiLSTM classifier.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_filters: Number of convolutional filters
            kernel_size: Kernel size for convolution
            pool_size: Max pooling size
            lstm_hidden_dim: LSTM hidden dimension
            dropout_conv: Dropout after conv layer
            dropout_lstm: Dropout after LSTM
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
        self.pool_size = pool_size
        self.lstm_hidden_dim = lstm_hidden_dim

        # Convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same'
        )

        # Max pooling
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Output layer (BiLSTM output is 2 * hidden_dim)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

        # Dropout layers
        self.dropout_conv = nn.Dropout(dropout_conv)
        self.dropout_lstm = nn.Dropout(dropout_lstm)

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

        # Convolution + ReLU + Dropout
        conv_out = self.dropout_conv(F.relu(self.conv1d(embedded)))

        # Max pooling
        pooled = self.maxpool(conv_out)

        # Prepare for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        pooled = pooled.permute(0, 2, 1)

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(pooled)

        # Concatenate final forward and backward hidden states
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)

        # Output
        output = self.fc(self.dropout_lstm(hidden_concat))

        return output.squeeze(-1)
