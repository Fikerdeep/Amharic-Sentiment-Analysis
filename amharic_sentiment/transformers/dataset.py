"""
Dataset utilities for transformer models.

This module provides a PyTorch Dataset class designed for use with
Hugging Face transformer tokenizers.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
from transformers import PreTrainedTokenizer

from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline


class TransformerDataset(Dataset):
    """
    PyTorch Dataset for transformer models.

    Handles tokenization with Hugging Face tokenizers and creates
    proper input format for transformer models.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        >>> dataset = TransformerDataset(texts, labels, tokenizer)
        >>> dataloader = DataLoader(dataset, batch_size=16)
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 128,
        preprocess: bool = True,
        return_token_type_ids: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text samples
            labels: List of labels (optional)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            preprocess: Whether to apply Amharic preprocessing
            return_token_type_ids: Whether to return token type IDs
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_token_type_ids = return_token_type_ids

        # Apply Amharic-specific preprocessing
        if preprocess:
            pipeline = PreprocessingPipeline()
            self.texts = [pipeline.process(t) for t in self.texts]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if self.return_token_type_ids and 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def create_transformer_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    tokenizer: PreTrainedTokenizer,
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    test_texts: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None,
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for transformer training.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        tokenizer: Hugging Face tokenizer
        val_texts: Validation texts
        val_labels: Validation labels
        test_texts: Test texts
        test_labels: Test labels
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}

    # Training
    train_dataset = TransformerDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Validation
    if val_texts is not None:
        val_dataset = TransformerDataset(
            val_texts, val_labels, tokenizer, max_length
        )
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    # Test
    if test_texts is not None:
        test_dataset = TransformerDataset(
            test_texts, test_labels, tokenizer, max_length
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return dataloaders
