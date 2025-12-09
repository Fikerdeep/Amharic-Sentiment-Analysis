"""
Amharic Sentiment Analysis Package

A deep learning-based sentiment analysis system for Amharic text.
Supports multiple model architectures: CNN, BiLSTM, GRU, and CNN-BiLSTM.
"""

__version__ = "1.0.0"
__author__ = "Fikerte Shalemo"

from amharic_sentiment.preprocessing.text_cleaner import AmharicTextCleaner
from amharic_sentiment.preprocessing.normalizer import AmharicNormalizer
from amharic_sentiment.data.dataset import AmharicSentimentDataset
from amharic_sentiment.models import CNN, BiLSTM, GRU, CNNBiLSTM

__all__ = [
    "AmharicTextCleaner",
    "AmharicNormalizer",
    "AmharicSentimentDataset",
    "CNN",
    "BiLSTM",
    "GRU",
    "CNNBiLSTM",
]
