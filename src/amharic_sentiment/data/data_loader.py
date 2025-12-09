"""
Data loading utilities for Amharic sentiment analysis.

This module provides convenience functions for loading and preparing
data from various sources.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from amharic_sentiment.data.dataset import AmharicSentimentDataset


def load_data(
    positive_file: Optional[str] = None,
    negative_file: Optional[str] = None,
    csv_file: Optional[str] = None,
    combined_file: Optional[str] = None,
    max_words: int = 15000,
    max_len: int = 20,
    **kwargs
) -> AmharicSentimentDataset:
    """
    Load data from various file formats.

    Args:
        positive_file: Path to positive samples file
        negative_file: Path to negative samples file
        csv_file: Path to CSV file
        combined_file: Path to combined file (label<tab>text format)
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
        **kwargs: Additional arguments for specific loaders

    Returns:
        Loaded AmharicSentimentDataset
    """
    dataset = AmharicSentimentDataset(max_words=max_words, max_len=max_len)

    if positive_file and negative_file:
        dataset.load_from_files(positive_file, negative_file, **kwargs)
    elif csv_file:
        dataset.load_from_csv(csv_file, **kwargs)
    elif combined_file:
        dataset.load_from_combined_file(combined_file, **kwargs)
    else:
        raise ValueError(
            "Must provide either (positive_file and negative_file), "
            "csv_file, or combined_file"
        )

    return dataset


def create_data_splits(
    dataset: AmharicSentimentDataset,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Create train/validation/test splits from a dataset.

    Args:
        dataset: The AmharicSentimentDataset to split
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing split data and metadata
    """
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'vocab_size': dataset.vocab_size,
        'tokenizer': dataset.tokenizer,
        'stats': dataset.get_stats()
    }


def load_embeddings(
    filepath: str,
    embedding_dim: int = 100
) -> Dict[str, np.ndarray]:
    """
    Load pre-trained word embeddings from file.

    Args:
        filepath: Path to embeddings file (Word2Vec format)
        embedding_dim: Expected embedding dimension

    Returns:
        Dictionary mapping words to embedding vectors
    """
    embeddings = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            values = line.strip().split()

            # Skip header line if present
            if line_num == 0 and len(values) == 2:
                continue

            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    embeddings[word] = vector
            except ValueError:
                continue

    return embeddings


def create_embedding_matrix(
    tokenizer,
    embeddings: Dict[str, np.ndarray],
    vocab_size: int,
    embedding_dim: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Create embedding matrix from tokenizer and pre-trained embeddings.

    Args:
        tokenizer: Fitted Keras tokenizer
        embeddings: Dictionary of word embeddings
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings

    Returns:
        Tuple of (embedding_matrix, coverage_ratio)
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_count = 0

    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue

        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            found_count += 1

    coverage = found_count / min(len(tokenizer.word_index), vocab_size)

    return embedding_matrix, coverage
