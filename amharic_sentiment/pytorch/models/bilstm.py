"""
PyTorch Bidirectional LSTM model for Amharic sentiment analysis.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from amharic_sentiment.pytorch.models.base import BaseClassifier


class BiLSTMClassifier(BaseClassifier):
    """
    Bidirectional LSTM for sentiment classification.

    Architecture:
        - Embedding layer
        - Bidirectional LSTM (2 layers)
        - Fully connected layers with dropout
        - Sigmoid output

    Example:
        >>> model = BiLSTMClassifier(vocab_size=10000, embedding_dim=32)
        >>> output = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        fc_hidden_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 1,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize BiLSTM classifier.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            fc_hidden_dim: Fully connected hidden dimension
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

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        # BiLSTM output is 2 * hidden_dim (forward + backward)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)

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
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate final forward and backward hidden states
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        hidden_forward = hidden[-2, :, :]  # Last forward layer
        hidden_backward = hidden[-1, :, :]  # Last backward layer
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)

        # Fully connected layers
        hidden = self.dropout(torch.relu(self.fc1(hidden_concat)))
        output = self.fc2(hidden)

        return output.squeeze(-1)
