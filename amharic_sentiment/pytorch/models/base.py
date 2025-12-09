"""
Base PyTorch model class for sentiment analysis.

This module provides an abstract base class for all PyTorch sentiment
classification models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseClassifier(nn.Module, ABC):
    """
    Abstract base class for PyTorch sentiment classifiers.

    All model architectures should inherit from this class and implement
    the forward method.

    Attributes:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        num_classes: Number of output classes (1 for binary)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_classes: int = 1,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize the base classifier.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            padding_idx: Index used for padding
            pretrained_embeddings: Pre-trained embedding matrix
            freeze_embeddings: Whether to freeze embedding weights
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        # Create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (sigmoid applied for binary classification).

        Args:
            x: Input tensor

        Returns:
            Prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                return torch.sigmoid(logits)
            else:
                return torch.softmax(logits, dim=-1)

    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }

    def summary(self) -> str:
        """Get model summary."""
        params = self.get_num_parameters()
        lines = [
            f"Model: {self.__class__.__name__}",
            f"Total parameters: {params['total']:,}",
            f"Trainable parameters: {params['trainable']:,}",
            f"Non-trainable parameters: {params['non_trainable']:,}",
            "",
            "Layers:",
        ]

        for name, module in self.named_modules():
            if name:
                lines.append(f"  {name}: {module.__class__.__name__}")

        return "\n".join(lines)
