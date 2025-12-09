"""
Training pipeline for Amharic sentiment analysis models.

This module provides a high-level training interface that handles
the complete training workflow including data preparation, model training,
and evaluation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union, Tuple
import numpy as np

from tensorflow.keras.callbacks import History

from amharic_sentiment.models.base import BaseModel
from amharic_sentiment.models import CNN, BiLSTM, GRU, CNNBiLSTM
from amharic_sentiment.data.dataset import AmharicSentimentDataset
from amharic_sentiment.training.callbacks import get_callbacks, TrainingProgressCallback
from amharic_sentiment.evaluation.metrics import evaluate_model


# Model registry
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'cnn': CNN,
    'bilstm': BiLSTM,
    'gru': GRU,
    'cnn_bilstm': CNNBiLSTM,
    'cnn-bilstm': CNNBiLSTM
}


class Trainer:
    """
    High-level trainer for sentiment analysis models.

    Handles the complete training workflow including:
    - Data loading and preparation
    - Model creation and configuration
    - Training with callbacks
    - Evaluation and logging

    Example:
        >>> trainer = Trainer(
        ...     model_type='cnn_bilstm',
        ...     dataset=dataset,
        ...     output_dir='experiments/exp1'
        ... )
        >>> trainer.train(epochs=10, batch_size=32)
        >>> results = trainer.evaluate()
    """

    def __init__(
        self,
        model_type: str,
        dataset: AmharicSentimentDataset,
        output_dir: str = "experiments",
        experiment_name: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize the trainer.

        Args:
            model_type: Type of model ('cnn', 'bilstm', 'gru', 'cnn_bilstm')
            dataset: Prepared AmharicSentimentDataset
            output_dir: Directory for saving outputs
            experiment_name: Name for this experiment
            model_params: Additional parameters for model initialization
            embedding_matrix: Pre-trained embedding matrix
        """
        self.model_type = model_type.lower()
        self.dataset = dataset
        self.embedding_matrix = embedding_matrix
        self.model_params = model_params or {}

        # Set up experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.model_type}_{timestamp}"

        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model: Optional[BaseModel] = None
        self.history: Optional[History] = None
        self.data_splits: Optional[Dict[str, np.ndarray]] = None

        # Training state
        self._is_trained = False

    def _create_model(self) -> BaseModel:
        """Create and return a model instance."""
        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_class = MODEL_REGISTRY[self.model_type]

        # Default parameters
        params = {
            'vocab_size': self.dataset.vocab_size,
            'max_len': self.dataset.max_len,
            'embedding_matrix': self.embedding_matrix
        }

        # Override with user parameters
        params.update(self.model_params)

        return model_class(**params)

    def prepare_data(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data splits for training.

        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed

        Returns:
            Dictionary with data splits
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.dataset.prepare_data(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )

        self.data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        return self.data_splits

    def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        use_class_weights: bool = False,
        callbacks_config: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ) -> History:
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_class_weights: Whether to use class weights
            callbacks_config: Configuration for callbacks
            verbose: Verbosity level

        Returns:
            Training history
        """
        # Prepare data if not already done
        if self.data_splits is None:
            self.prepare_data()

        # Create and compile model
        self.model = self._create_model()
        self.model.build()
        self.model.compile(learning_rate=learning_rate)

        if verbose:
            self.model.summary()

        # Set up callbacks
        callbacks_config = callbacks_config or {}
        callbacks = get_callbacks(
            model_name=self.experiment_name,
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            log_dir=str(self.output_dir / "logs"),
            **callbacks_config
        )
        callbacks.append(TrainingProgressCallback())

        # Class weights
        class_weight = None
        if use_class_weights:
            class_weight = self.dataset.get_class_weights()

        # Prepare validation data
        validation_data = None
        if self.data_splits['X_val'] is not None:
            validation_data = (
                self.data_splits['X_val'],
                self.data_splits['y_val']
            )

        # Train
        self.history = self.model.fit(
            self.data_splits['X_train'],
            self.data_splits['y_train'],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose
        )

        self._is_trained = True

        # Save training config
        self._save_config(epochs, batch_size, learning_rate, use_class_weights)

        return self.history

    def evaluate(self, verbose: int = 1) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            verbose: Verbosity level

        Returns:
            Dictionary with evaluation metrics
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        results = evaluate_model(
            self.model,
            self.data_splits['X_test'],
            self.data_splits['y_test']
        )

        if verbose:
            print("\n" + "=" * 50)
            print("Evaluation Results")
            print("=" * 50)
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in results.items()
                if not isinstance(v, np.ndarray)
            }
            json.dump(serializable_results, f, indent=2)

        return results

    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model.

        Args:
            filename: Custom filename for the model

        Returns:
            Path to saved model
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if filename is None:
            filename = f"{self.experiment_name}_final.keras"

        model_path = self.output_dir / filename
        self.model.save(str(model_path))

        return str(model_path)

    def _save_config(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        use_class_weights: bool
    ) -> None:
        """Save training configuration."""
        config = {
            'model_type': self.model_type,
            'experiment_name': self.experiment_name,
            'model_params': self.model_params,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'use_class_weights': use_class_weights
            },
            'dataset_stats': self.dataset.get_stats(),
            'model_config': self.model.get_config()
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def train_model(
    model_type: str,
    positive_file: str,
    negative_file: str,
    output_dir: str = "experiments",
    epochs: int = 10,
    batch_size: int = 32,
    max_words: int = 15000,
    max_len: int = 20,
    **kwargs
) -> Tuple[BaseModel, Dict[str, Any]]:
    """
    Convenience function for training a model from files.

    Args:
        model_type: Type of model to train
        positive_file: Path to positive samples
        negative_file: Path to negative samples
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
        **kwargs: Additional arguments for Trainer

    Returns:
        Tuple of (trained model, evaluation results)
    """
    # Load data
    dataset = AmharicSentimentDataset(max_words=max_words, max_len=max_len)
    dataset.load_from_files(positive_file, negative_file)

    # Create trainer and train
    trainer = Trainer(
        model_type=model_type,
        dataset=dataset,
        output_dir=output_dir,
        **kwargs
    )

    trainer.train(epochs=epochs, batch_size=batch_size)
    results = trainer.evaluate()

    return trainer.model, results
