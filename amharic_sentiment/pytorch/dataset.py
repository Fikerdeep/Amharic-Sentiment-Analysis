"""
PyTorch Dataset for Amharic sentiment analysis.

This module provides a PyTorch Dataset class that wraps the preprocessing
and tokenization logic for use with PyTorch DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline


class AmharicDataset(Dataset):
    """
    PyTorch Dataset for Amharic sentiment data.

    Example:
        >>> dataset = AmharicDataset(texts, labels, tokenizer, max_len=20)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_len: int = 20,
        preprocess: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text samples
            labels: List of labels (optional for inference)
            tokenizer: Fitted tokenizer (Keras or custom)
            max_len: Maximum sequence length
            preprocess: Whether to apply preprocessing
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess = preprocess

        if preprocess:
            self.pipeline = PreprocessingPipeline()
            self.texts = [self.pipeline.process(t) for t in self.texts]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize
        if self.tokenizer is not None:
            sequence = self.tokenizer.texts_to_sequences([text])[0]
        else:
            # Simple word-to-index if no tokenizer
            sequence = [hash(w) % 10000 for w in text.split()]

        # Pad/truncate
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]

        item = {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'text': text
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def create_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    test_texts: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    tokenizer=None,
    max_len: int = 20,
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        test_texts: Test texts
        test_labels: Test labels
        tokenizer: Fitted tokenizer
        max_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}

    # Training
    train_dataset = AmharicDataset(train_texts, train_labels, tokenizer, max_len)
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Validation
    if val_texts is not None:
        val_dataset = AmharicDataset(val_texts, val_labels, tokenizer, max_len)
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    # Test
    if test_texts is not None:
        test_dataset = AmharicDataset(test_texts, test_labels, tokenizer, max_len)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return dataloaders
