"""
Model service for loading and managing sentiment analysis models.

Handles model loading, caching, and inference for different model types.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


class BaseModelService(ABC):
    """Abstract base class for model services."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> bool:
        """Load the model from path."""
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Make predictions on texts."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class TensorFlowModelService(BaseModelService):
    """Service for TensorFlow/Keras models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_len = 20
        self.pipeline = None

    def load(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        max_len: int = 20
    ) -> bool:
        """
        Load TensorFlow model and tokenizer.

        Args:
            model_path: Path to saved Keras model
            tokenizer_path: Path to pickled tokenizer
            max_len: Maximum sequence length

        Returns:
            True if loaded successfully
        """
        try:
            from tensorflow.keras.models import load_model
            from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline

            self.model = load_model(model_path)
            self.max_len = max_len
            self.pipeline = PreprocessingPipeline()

            if tokenizer_path:
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)

            logger.info(f"TensorFlow model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            return False

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Make predictions."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # Preprocess
        processed = [self.pipeline.process(t) for t in texts]

        # Tokenize
        sequences = self.tokenizer.texts_to_sequences(processed)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        # Predict
        probabilities = self.model.predict(padded, verbose=0).flatten()

        results = []
        for prob in probabilities:
            sentiment = "positive" if prob > 0.5 else "negative"
            results.append((sentiment, float(prob)))

        return results

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class PyTorchModelService(BaseModelService):
    """Service for PyTorch models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_len = 20
        self.pipeline = None

    def load(
        self,
        model_path: str,
        model_class: str = "CNNBiLSTMClassifier",
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 15000,
        max_len: int = 20,
        device: str = "cpu",
        **model_kwargs
    ) -> bool:
        """
        Load PyTorch model.

        Args:
            model_path: Path to saved model state dict
            model_class: Model class name
            tokenizer_path: Path to tokenizer
            vocab_size: Vocabulary size
            max_len: Maximum sequence length
            device: Device to use
            **model_kwargs: Additional model arguments

        Returns:
            True if loaded successfully
        """
        try:
            import torch
            from amharic_sentiment.pytorch import models as pytorch_models
            from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline

            self.device = torch.device(device)
            self.max_len = max_len
            self.pipeline = PreprocessingPipeline()

            # Get model class
            model_cls = getattr(pytorch_models, model_class)
            self.model = model_cls(vocab_size=vocab_size, **model_kwargs)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            if tokenizer_path:
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)

            logger.info(f"PyTorch model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Make predictions."""
        import torch

        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # Preprocess
        processed = [self.pipeline.process(t) for t in texts]

        # Tokenize
        sequences = self.tokenizer.texts_to_sequences(processed)

        # Pad
        padded = []
        for seq in sequences:
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[:self.max_len]
            padded.append(seq)

        # Convert to tensor
        input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()

        results = []
        for prob in probabilities:
            sentiment = "positive" if prob > 0.5 else "negative"
            results.append((sentiment, float(prob)))

        return results

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class TransformerModelService(BaseModelService):
    """Service for Transformer models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 128
        self.pipeline = None

    def load(
        self,
        model_path: str,
        model_name: str = "xlm-roberta",
        max_length: int = 128,
        device: str = "cpu"
    ) -> bool:
        """
        Load Transformer model.

        Args:
            model_path: Path to saved model
            model_name: Base model name
            max_length: Maximum sequence length
            device: Device to use

        Returns:
            True if loaded successfully
        """
        try:
            import torch
            from transformers import AutoTokenizer
            from amharic_sentiment.transformers.models import TransformerClassifier
            from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline

            self.device = torch.device(device)
            self.max_length = max_length
            self.pipeline = PreprocessingPipeline()

            # Load model
            self.model = TransformerClassifier.load_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            # Load tokenizer
            self.tokenizer = self.model.get_tokenizer()

            logger.info(f"Transformer model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            return False

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Make predictions."""
        import torch

        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # Preprocess
        processed = [self.pipeline.process(t) for t in texts]

        # Tokenize
        encodings = self.tokenizer(
            processed,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()

        results = []
        for prob in probabilities:
            sentiment = "positive" if prob > 0.5 else "negative"
            results.append((sentiment, float(prob)))

        return results

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class ModelManager:
    """
    Manager for loading and switching between different model types.

    Singleton pattern to ensure only one model is loaded at a time.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.services: Dict[str, BaseModelService] = {
            'tensorflow': TensorFlowModelService(),
            'pytorch': PyTorchModelService(),
            'transformer': TransformerModelService()
        }
        self.current_type: Optional[str] = None
        self._initialized = True

    def load_model(self, model_type: str, **kwargs) -> bool:
        """
        Load a model of specified type.

        Args:
            model_type: Type of model ('tensorflow', 'pytorch', 'transformer')
            **kwargs: Arguments passed to the service's load method

        Returns:
            True if loaded successfully
        """
        if model_type not in self.services:
            raise ValueError(f"Unknown model type: {model_type}")

        success = self.services[model_type].load(**kwargs)
        if success:
            self.current_type = model_type

        return success

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Make predictions using the current model."""
        if self.current_type is None:
            raise RuntimeError("No model loaded")

        return self.services[self.current_type].predict(texts)

    def is_loaded(self) -> bool:
        """Check if any model is loaded."""
        if self.current_type is None:
            return False
        return self.services[self.current_type].is_loaded()

    def get_current_type(self) -> Optional[str]:
        """Get the current model type."""
        return self.current_type


# Global model manager instance
model_manager = ModelManager()
