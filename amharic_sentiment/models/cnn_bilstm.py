"""
CNN-BiLSTM hybrid model for Amharic sentiment analysis.

This module implements a hybrid architecture combining CNN for local
feature extraction and BiLSTM for sequence modeling.
"""

from typing import Optional
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout
)

from amharic_sentiment.models.base import BaseModel


class CNNBiLSTM(BaseModel):
    """
    Hybrid CNN-BiLSTM for sentiment analysis.

    This architecture combines:
    - CNN for extracting local n-gram features
    - BiLSTM for capturing long-range dependencies

    Architecture:
        - Embedding layer
        - Conv1D with ReLU activation
        - Dropout
        - MaxPooling1D
        - Bidirectional LSTM
        - Dropout
        - Dense output with sigmoid

    Example:
        >>> model = CNNBiLSTM(vocab_size=10000, embedding_dim=32, max_len=20)
        >>> model.build()
        >>> model.compile()
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        max_len: int = 20,
        embedding_matrix: Optional[np.ndarray] = None,
        trainable_embeddings: bool = True,
        filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 4,
        lstm_units: int = 64,
        dropout_rate_conv: float = 0.2,
        dropout_rate_lstm: float = 0.3
    ):
        """
        Initialize the CNN-BiLSTM model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            max_len: Maximum sequence length
            embedding_matrix: Pre-trained embedding matrix
            trainable_embeddings: Whether to train embeddings
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            pool_size: Size of max pooling window
            lstm_units: Units in BiLSTM layer
            dropout_rate_conv: Dropout rate after conv layer
            dropout_rate_lstm: Dropout rate after LSTM layer
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            trainable_embeddings=trainable_embeddings,
            name="CNN-BiLSTM"
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dropout_rate_conv = dropout_rate_conv
        self.dropout_rate_lstm = dropout_rate_lstm

    def build(self) -> Model:
        """
        Build the CNN-BiLSTM model.

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

        model.add(Dropout(self.dropout_rate_conv, name='dropout_conv'))

        # Max pooling
        model.add(MaxPooling1D(pool_size=self.pool_size, name='max_pool'))

        # Bidirectional LSTM
        model.add(Bidirectional(
            LSTM(self.lstm_units),
            name='bilstm'
        ))

        model.add(Dropout(self.dropout_rate_lstm, name='dropout_lstm'))

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
            'pool_size': self.pool_size,
            'lstm_units': self.lstm_units,
            'dropout_rate_conv': self.dropout_rate_conv,
            'dropout_rate_lstm': self.dropout_rate_lstm
        })
        return config
