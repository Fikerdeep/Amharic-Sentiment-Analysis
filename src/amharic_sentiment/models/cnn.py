"""
CNN model for Amharic sentiment analysis.

This module implements a Convolutional Neural Network architecture
for text classification.
"""

from typing import Optional
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout
)

from amharic_sentiment.models.base import BaseModel


class CNN(BaseModel):
    """
    Convolutional Neural Network for sentiment analysis.

    Architecture:
        - Embedding layer
        - Conv1D layer with ReLU activation
        - GlobalMaxPooling1D
        - Dense layers with dropout
        - Sigmoid output

    Example:
        >>> model = CNN(vocab_size=10000, embedding_dim=100, max_len=20)
        >>> model.build()
        >>> model.compile()
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        max_len: int = 20,
        embedding_matrix: Optional[np.ndarray] = None,
        trainable_embeddings: bool = True,
        filters: int = 64,
        kernel_size: int = 3,
        dense_units: int = 64,
        dropout_rate: float = 0.2,
        dropout_rate_dense: float = 0.3
    ):
        """
        Initialize the CNN model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            max_len: Maximum sequence length
            embedding_matrix: Pre-trained embedding matrix
            trainable_embeddings: Whether to train embeddings
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate after conv layer
            dropout_rate_dense: Dropout rate after dense layer
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            trainable_embeddings=trainable_embeddings,
            name="CNN"
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.dropout_rate_dense = dropout_rate_dense

    def build(self) -> Model:
        """
        Build the CNN model.

        Returns:
            Compiled Keras Model
        """
        model = Sequential(name=self.name)

        # Embedding layer
        if self.embedding_matrix is not None:
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[self.embedding_matrix],
                trainable=self.trainable_embeddings,
                name='embedding'
            ))
        else:
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                trainable=self.trainable_embeddings,
                name='embedding'
            ))

        # Convolutional layer
        model.add(Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu',
            name='conv1d'
        ))

        model.add(Dropout(self.dropout_rate, name='dropout_conv'))

        # Global max pooling
        model.add(GlobalMaxPooling1D(name='global_max_pool'))

        # Dense layers
        model.add(Dense(self.dense_units, activation='relu', name='dense'))
        model.add(Dropout(self.dropout_rate_dense, name='dropout_dense'))

        # Output layer
        model.add(Dense(1, activation='sigmoid', name='output'))

        self.model = model
        return model

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'dropout_rate_dense': self.dropout_rate_dense
        })
        return config
