"""
Transformer models for Amharic sentiment analysis.

This module provides wrappers around Hugging Face transformer models
for sequence classification.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)


# Recommended models for Amharic
AMHARIC_MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlm-roberta': 'xlm-roberta-base',
    'xlm-roberta-large': 'xlm-roberta-large',
    'distilbert-multi': 'distilbert-base-multilingual-cased',
    'afro-xlmr': 'Davlan/afro-xlmr-base',  # African languages focused
}


def get_model_for_amharic(
    model_name: str = 'xlm-roberta',
    num_labels: int = 2,
    from_pretrained: bool = True
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Get a pre-trained model suitable for Amharic text.

    Args:
        model_name: Model identifier (see AMHARIC_MODELS)
        num_labels: Number of classification labels
        from_pretrained: Whether to load pre-trained weights

    Returns:
        Tuple of (model, tokenizer)
    """
    # Resolve model name
    if model_name in AMHARIC_MODELS:
        model_path = AMHARIC_MODELS[model_name]
    else:
        model_path = model_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    if from_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        )
    else:
        config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(config)

    return model, tokenizer


class TransformerClassifier(nn.Module):
    """
    Custom transformer classifier with additional layers.

    Wraps a pre-trained transformer with custom classification head
    for more flexibility.

    Example:
        >>> classifier = TransformerClassifier('xlm-roberta')
        >>> outputs = classifier(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = 'xlm-roberta',
        num_labels: int = 1,
        dropout: float = 0.1,
        freeze_base: bool = False,
        freeze_layers: Optional[int] = None
    ):
        """
        Initialize the transformer classifier.

        Args:
            model_name: Pre-trained model name
            num_labels: Number of output labels
            dropout: Dropout rate for classification head
            freeze_base: Whether to freeze the entire base model
            freeze_layers: Number of layers to freeze (from bottom)
        """
        super().__init__()

        # Resolve model path
        if model_name in AMHARIC_MODELS:
            model_path = AMHARIC_MODELS[model_name]
        else:
            model_path = model_name

        self.model_name = model_name
        self.num_labels = num_labels

        # Load base model
        self.transformer = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.transformer.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Freeze layers if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif freeze_layers is not None:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers: int):
        """Freeze bottom N layers of the transformer."""
        # Freeze embeddings
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.transformer, 'encoder'):
            for i, layer in enumerate(self.transformer.encoder.layer):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT)

        Returns:
            Classification logits
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits.squeeze(-1) if self.num_labels == 1 else logits

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer for this model."""
        if self.model_name in AMHARIC_MODELS:
            model_path = AMHARIC_MODELS[self.model_name]
        else:
            model_path = self.model_name
        return AutoTokenizer.from_pretrained(model_path)

    def save_pretrained(self, path: str):
        """Save model and config."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'hidden_size': self.hidden_size
        }, path)

    @classmethod
    def load_pretrained(cls, path: str, **kwargs) -> 'TransformerClassifier':
        """Load a saved model."""
        checkpoint = torch.load(path)
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels'],
            **kwargs
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
