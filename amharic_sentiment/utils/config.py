"""
Configuration management for Amharic sentiment analysis.

This module provides utilities for loading and managing configuration
from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration."""
    positive_file: str = "dataset/postive comment.txt"
    negative_file: str = "dataset/negative comment.txt"
    embeddings_file: Optional[str] = None
    max_words: int = 15000
    max_len: int = 20
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    encoding: str = "utf-8"


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "cnn_bilstm"
    embedding_dim: int = 32
    trainable_embeddings: bool = True

    # CNN parameters
    filters: int = 64
    kernel_size: int = 3

    # LSTM/GRU parameters
    lstm_units: int = 64
    gru_units: int = 64

    # Common parameters
    dropout_rate: float = 0.3
    dense_units: int = 64


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    use_class_weights: bool = False

    # Callbacks
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    min_lr: float = 1e-7


@dataclass
class Config:
    """Main configuration class combining all configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Paths
    output_dir: str = "experiments"
    checkpoint_dir: str = "saved_models"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            output_dir=config_dict.get('output_dir', 'experiments'),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'saved_models'),
            log_dir=config_dict.get('log_dir', 'logs')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': {
                'positive_file': self.data.positive_file,
                'negative_file': self.data.negative_file,
                'embeddings_file': self.data.embeddings_file,
                'max_words': self.data.max_words,
                'max_len': self.data.max_len,
                'test_size': self.data.test_size,
                'val_size': self.data.val_size,
                'random_state': self.data.random_state,
                'encoding': self.data.encoding
            },
            'model': {
                'model_type': self.model.model_type,
                'embedding_dim': self.model.embedding_dim,
                'trainable_embeddings': self.model.trainable_embeddings,
                'filters': self.model.filters,
                'kernel_size': self.model.kernel_size,
                'lstm_units': self.model.lstm_units,
                'gru_units': self.model.gru_units,
                'dropout_rate': self.model.dropout_rate,
                'dense_units': self.model.dense_units
            },
            'training': {
                'epochs': self.training.epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'optimizer': self.training.optimizer,
                'loss': self.training.loss,
                'use_class_weights': self.training.use_class_weights,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience,
                'min_lr': self.training.min_lr
            },
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }

    def save(self, filepath: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            filepath: Path to save the configuration
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def update(self, **kwargs) -> 'Config':
        """
        Update configuration with new values.

        Args:
            **kwargs: Key-value pairs to update

        Returns:
            Updated Config instance
        """
        config_dict = self.to_dict()

        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'model.embedding_dim'
                parts = key.split('.')
                d = config_dict
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = value
            else:
                config_dict[key] = value

        return Config.from_dict(config_dict)


def load_config(filepath: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        filepath: Optional path to configuration file

    Returns:
        Config instance
    """
    if filepath and Path(filepath).exists():
        return Config.from_yaml(filepath)
    return Config()


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
