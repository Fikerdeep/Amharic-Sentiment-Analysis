"""
PyTorch GRU model for Amharic sentiment analysis.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from amharic_sentiment.pytorch.models.base import BaseClassifier


class GRUClassifier(BaseClassifier):
    """
    GRU (Gated Recurrent Unit) for sentiment classification.

    Architecture:
        - Embedding layer
        - Stacked GRU layers
        - Fully connected output layer
        - Sigmoid output

    Example:
        >>> model = GRUClassifier(vocab_size=10000, embedding_dim=100)
        >>> output = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 1,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize GRU classifier.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
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

        # Stacked GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

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

        # GRU
        gru_out, hidden = self.gru(embedded)

        # Take the last hidden state from the final layer
        # hidden shape: (num_layers, batch, hidden_dim)
        final_hidden = hidden[-1, :, :]

        # Output
        output = self.fc(self.dropout(final_hidden))

        return output.squeeze(-1)
