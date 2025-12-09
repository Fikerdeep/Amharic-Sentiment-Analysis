"""
Base model class for sentiment analysis models.

This module provides an abstract base class that defines the interface
for all sentiment analysis model architectures.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import History


class BaseModel(ABC):
    """
    Abstract base class for sentiment analysis models.

    All model architectures should inherit from this class and implement
    the required methods.

    Attributes:
        model: The underlying Keras model
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        max_len: Maximum sequence length
        name: Name of the model architecture
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        max_len: int = 20,
        embedding_matrix: Optional[np.ndarray] = None,
        trainable_embeddings: bool = True,
        name: str = "BaseModel"
    ):
        """
        Initialize the base model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            max_len: Maximum sequence length
            embedding_matrix: Pre-trained embedding matrix (optional)
            trainable_embeddings: Whether to train embeddings
            name: Name of the model
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.name = name
        self.model: Optional[Model] = None
        self._history: Optional[History] = None

    @abstractmethod
    def build(self) -> Model:
        """
        Build and return the Keras model.

        Returns:
            Compiled Keras Model
        """
        pass

    def compile(
        self,
        optimizer: str = 'adam',
        loss: str = 'binary_crossentropy',
        metrics: Optional[list] = None,
        learning_rate: float = 0.001
    ) -> 'BaseModel':
        """
        Compile the model.

        Args:
            optimizer: Optimizer name or instance
            loss: Loss function
            metrics: List of metrics to track
            learning_rate: Learning rate for optimizer

        Returns:
            Self for method chaining
        """
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.metrics import Precision, Recall

        if self.model is None:
            self.build()

        if metrics is None:
            metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]

        if optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        class_weight: Optional[Dict[int, float]] = None,
        verbose: int = 1
    ) -> History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            class_weight: Class weights for imbalanced data
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        self._history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose
        )

        return self._history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        probabilities = self.model.predict(X)
        return (probabilities > threshold).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        return self.model.predict(X).flatten()

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Verbosity level

        Returns:
            Dictionary of metric names and values
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        metric_names = self.model.metrics_names

        return {name: value for name, value in zip(metric_names, results)}

    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        self.model.summary()

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        self.model.save(filepath)

    def load(self, filepath: str) -> 'BaseModel':
        """
        Load model weights from a file.

        Args:
            filepath: Path to the saved model

        Returns:
            Self for method chaining
        """
        self.model = load_model(filepath)
        return self

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary of model configuration
        """
        return {
            'name': self.name,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_len': self.max_len,
            'trainable_embeddings': self.trainable_embeddings
        }

    @property
    def history(self) -> Optional[History]:
        """Get training history."""
        return self._history

    def __repr__(self) -> str:
        return f"{self.name}(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, max_len={self.max_len})"
