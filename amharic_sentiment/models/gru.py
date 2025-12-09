"""
GRU model for Amharic sentiment analysis.

This module implements a Gated Recurrent Unit architecture
for text classification.
"""

from typing import Optional
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, GRU as KerasGRU,
    Dense, Dropout
)

from amharic_sentiment.models.base import BaseModel


class GRUModel(BaseModel):
    """
    GRU (Gated Recurrent Unit) for sentiment analysis.

    Architecture:
        - Embedding layer
        - GRU layer (return sequences)
        - GRU layer
        - Dense output with sigmoid

    Example:
        >>> model = GRUModel(vocab_size=10000, embedding_dim=100, max_len=20)
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
        gru_units_1: int = 64,
        gru_units_2: int = 32,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the GRU model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            max_len: Maximum sequence length
            embedding_matrix: Pre-trained embedding matrix
            trainable_embeddings: Whether to train embeddings
            gru_units_1: Units in first GRU layer
            gru_units_2: Units in second GRU layer
            dropout_rate: Dropout rate for GRU layers
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            trainable_embeddings=trainable_embeddings,
            name="GRU"
        )

        self.gru_units_1 = gru_units_1
        self.gru_units_2 = gru_units_2
        self.dropout_rate = dropout_rate

    def build(self) -> Model:
        """
        Build the GRU model.

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
                mask_zero=True,
                name='embedding'
            ))
        else:
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                trainable=self.trainable_embeddings,
                mask_zero=True,
                name='embedding'
            ))

        # First GRU layer
        model.add(KerasGRU(
            self.gru_units_1,
            dropout=self.dropout_rate,
            return_sequences=True,
            name='gru_1'
        ))

        # Second GRU layer
        model.add(KerasGRU(
            self.gru_units_2,
            dropout=self.dropout_rate,
            return_sequences=False,
            name='gru_2'
        ))

        # Output layer
        model.add(Dense(1, activation='sigmoid', name='output'))

        self.model = model
        return model

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'gru_units_1': self.gru_units_1,
            'gru_units_2': self.gru_units_2,
            'dropout_rate': self.dropout_rate
        })
        return config
