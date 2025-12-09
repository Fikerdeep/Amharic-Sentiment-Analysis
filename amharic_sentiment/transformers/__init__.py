"""
Hugging Face Transformers integration for Amharic sentiment analysis.

This module provides fine-tuning capabilities for multilingual
transformer models like mBERT and XLM-RoBERTa.
"""

from amharic_sentiment.transformers.models import (
    TransformerClassifier,
    get_model_for_amharic
)
from amharic_sentiment.transformers.dataset import TransformerDataset
from amharic_sentiment.transformers.trainer import TransformerTrainer

__all__ = [
    "TransformerClassifier",
    "TransformerDataset",
    "TransformerTrainer",
    "get_model_for_amharic"
]
