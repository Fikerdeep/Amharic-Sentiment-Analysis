"""
Bidirectional LSTM model for Amharic sentiment analysis.

This module implements a Bidirectional LSTM architecture
for text classification.
"""

from typing import Optional
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout
)

from amharic_sentiment.models.base import BaseModel


class BiLSTM(BaseModel):
    """
    Bidirectional LSTM for sentiment analysis.

    Architecture:
        - Embedding layer
        - Bidirectional LSTM (return sequences)
        - Dropout
        - Bidirectional LSTM
        - Dropout
        - Dense layer with ReLU
        - Dropout
        - Sigmoid output

    Example:
        >>> model = BiLSTM(vocab_size=10000, embedding_dim=32, max_len=20)
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
        lstm_units_1: int = 64,
        lstm_units_2: int = 32,
        dense_units: int = 64,
        dropout_rate_1: float = 0.3,
        dropout_rate_2: float = 0.2,
        dropout_rate_dense: float = 0.1
    ):
        """
        Initialize the BiLSTM model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            max_len: Maximum sequence length
            embedding_matrix: Pre-trained embedding matrix
            trainable_embeddings: Whether to train embeddings
            lstm_units_1: Units in first LSTM layer
            lstm_units_2: Units in second LSTM layer
            dense_units: Units in dense layer
            dropout_rate_1: Dropout rate after first LSTM
            dropout_rate_2: Dropout rate after second LSTM
            dropout_rate_dense: Dropout rate after dense layer
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            trainable_embeddings=trainable_embeddings,
            name="BiLSTM"
        )

        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.dropout_rate_dense = dropout_rate_dense

    def build(self) -> Model:
        """
        Build the BiLSTM model.

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

        # First Bidirectional LSTM
        model.add(Bidirectional(
            LSTM(self.lstm_units_1, return_sequences=True),
            name='bilstm_1'
        ))
        model.add(Dropout(self.dropout_rate_1, name='dropout_1'))

        # Second Bidirectional LSTM
        model.add(Bidirectional(
            LSTM(self.lstm_units_2),
            name='bilstm_2'
        ))
        model.add(Dropout(self.dropout_rate_2, name='dropout_2'))

        # Dense layer
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
            'lstm_units_1': self.lstm_units_1,
            'lstm_units_2': self.lstm_units_2,
            'dense_units': self.dense_units,
            'dropout_rate_1': self.dropout_rate_1,
            'dropout_rate_2': self.dropout_rate_2,
            'dropout_rate_dense': self.dropout_rate_dense
        })
        return config
