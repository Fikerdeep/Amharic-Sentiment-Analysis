"""
Dataset class for Amharic sentiment analysis.

This module provides a dataset class that handles loading, preprocessing,
tokenization, and preparation of Amharic text data for model training.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline


class AmharicSentimentDataset:
    """
    Dataset class for Amharic sentiment analysis.

    Handles data loading, preprocessing, tokenization, and preparation
    for model training and evaluation.

    Example:
        >>> dataset = AmharicSentimentDataset(max_words=5000, max_len=20)
        >>> dataset.load_from_files(
        ...     positive_file="data/positive.txt",
        ...     negative_file="data/negative.txt"
        ... )
        >>> X_train, X_test, y_train, y_test = dataset.prepare_data()
    """

    def __init__(
        self,
        max_words: int = 15000,
        max_len: int = 20,
        padding: str = 'post',
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None
    ):
        """
        Initialize the dataset.

        Args:
            max_words: Maximum number of words in vocabulary
            max_len: Maximum sequence length
            padding: Padding strategy ('pre' or 'post')
            preprocessing_pipeline: Custom preprocessing pipeline
        """
        self.max_words = max_words
        self.max_len = max_len
        self.padding = padding
        self.preprocessor = preprocessing_pipeline or PreprocessingPipeline()

        self.tokenizer: Optional[Tokenizer] = None
        self.texts: List[str] = []
        self.labels: np.ndarray = np.array([])
        self.vocab_size: int = 0

    def load_from_files(
        self,
        positive_file: str,
        negative_file: str,
        encoding: str = 'utf-8'
    ) -> 'AmharicSentimentDataset':
        """
        Load data from separate positive and negative files.

        Args:
            positive_file: Path to file with positive samples
            negative_file: Path to file with negative samples
            encoding: File encoding

        Returns:
            Self for method chaining
        """
        positive_texts = self._load_file(positive_file, encoding)
        negative_texts = self._load_file(negative_file, encoding)

        self.texts = positive_texts + negative_texts
        self.labels = np.array(
            [1] * len(positive_texts) + [0] * len(negative_texts)
        )

        return self

    def load_from_csv(
        self,
        filepath: str,
        text_column: str = 'review',
        label_column: str = 'sentiment',
        positive_label: str = 'positive',
        encoding: str = 'utf-16'
    ) -> 'AmharicSentimentDataset':
        """
        Load data from a CSV file.

        Args:
            filepath: Path to CSV file
            text_column: Name of the text column
            label_column: Name of the label column
            positive_label: Value representing positive sentiment
            encoding: File encoding

        Returns:
            Self for method chaining
        """
        import pandas as pd

        df = pd.read_csv(filepath, encoding=encoding)
        self.texts = df[text_column].tolist()
        self.labels = np.array([
            1 if label == positive_label else 0
            for label in df[label_column]
        ])

        return self

    def load_from_combined_file(
        self,
        filepath: str,
        encoding: str = 'utf-8'
    ) -> 'AmharicSentimentDataset':
        """
        Load data from a combined file with format: "label<tab>text".

        Args:
            filepath: Path to the combined file
            encoding: File encoding

        Returns:
            Self for method chaining
        """
        texts = []
        labels = []

        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t', 1)
                if len(parts) == 2:
                    label, text = parts
                    texts.append(text)
                    labels.append(1 if label.lower() in ['positive', '1', 'pos'] else 0)

        self.texts = texts
        self.labels = np.array(labels)

        return self

    def _load_file(self, filepath: str, encoding: str = 'utf-8') -> List[str]:
        """Load text from a file, one sample per line."""
        texts = []
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    def preprocess(self) -> 'AmharicSentimentDataset':
        """
        Apply preprocessing to all texts.

        Returns:
            Self for method chaining
        """
        self.texts = [self.preprocessor.process(text) for text in self.texts]
        return self

    def fit_tokenizer(self) -> 'AmharicSentimentDataset':
        """
        Fit the tokenizer on the texts.

        Returns:
            Self for method chaining
        """
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(self.texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        return self

    def tokenize(self, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Tokenize and pad text sequences.

        Args:
            texts: Texts to tokenize (uses self.texts if None)

        Returns:
            Padded sequences as numpy array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer() first.")

        texts_to_tokenize = texts if texts is not None else self.texts
        sequences = self.tokenizer.texts_to_sequences(texts_to_tokenize)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding=self.padding)

        return padded

    def prepare_data(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for training with train/val/test splits.

        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split

        # Preprocess and tokenize
        self.preprocess()
        self.fit_tokenizer()
        X = self.tokenize()
        y = self.labels

        # Split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

        # Split train+val into train and val
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_ratio,
                random_state=random_state, shuffle=shuffle
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_trainval, None, X_test, y_trainval, None, y_test

    def get_embedding_matrix(
        self,
        embeddings_path: str,
        embedding_dim: int = 100
    ) -> np.ndarray:
        """
        Create embedding matrix from pre-trained embeddings.

        Args:
            embeddings_path: Path to Word2Vec embeddings file
            embedding_dim: Dimension of embeddings

        Returns:
            Embedding matrix of shape (vocab_size, embedding_dim)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer() first.")

        # Load embeddings
        embeddings_dict = {}
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = vector

        # Create embedding matrix
        embedding_matrix = np.zeros((self.vocab_size, embedding_dim))
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data.

        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(self.labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.labels
        )
        return {i: w for i, w in zip(classes, weights)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        positive_count = np.sum(self.labels == 1)
        negative_count = np.sum(self.labels == 0)

        return {
            'total_samples': len(self.texts),
            'positive_samples': int(positive_count),
            'negative_samples': int(negative_count),
            'positive_ratio': float(positive_count / len(self.texts)) if self.texts else 0,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'max_words': self.max_words
        }

    def save_tokenizer(self, filepath: str) -> None:
        """Save the fitted tokenizer to a file."""
        import pickle

        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted.")

        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load_tokenizer(self, filepath: str) -> 'AmharicSentimentDataset':
        """Load a tokenizer from a file."""
        import pickle

        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.vocab_size = len(self.tokenizer.word_index) + 1
        return self
