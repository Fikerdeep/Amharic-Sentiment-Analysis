"""
Trainer for transformer models.

This module provides a specialized trainer for fine-tuning
Hugging Face transformer models on Amharic sentiment data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


class TransformerTrainer:
    """
    Trainer for fine-tuning transformer models.

    Features:
        - AdamW optimizer with weight decay
        - Linear/cosine learning rate warmup
        - Gradient clipping
        - Mixed precision training support
        - Early stopping
        - Best model checkpointing

    Example:
        >>> trainer = TransformerTrainer(model, tokenizer)
        >>> trainer.train(train_loader, val_loader, epochs=5)
        >>> results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        device: str = 'auto',
        output_dir: str = 'experiments',
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: Transformer model
            tokenizer: Model tokenizer (for saving)
            device: Device to use
            output_dir: Output directory
            experiment_name: Experiment name
        """
        self.model = model
        self.tokenizer = tokenizer

        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Setup experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"transformer_{timestamp}"

        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.best_val_loss = float('inf')

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        scheduler_type: str = 'linear',
        early_stopping_patience: int = 3,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup steps ratio
            max_grad_norm: Maximum gradient norm for clipping
            scheduler_type: 'linear' or 'cosine'
            early_stopping_patience: Early stopping patience
            verbose: Show progress

        Returns:
            Training history
        """
        # Setup optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = AdamW(optimizer_grouped_params, lr=learning_rate)

        # Setup scheduler
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        if scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, max_grad_norm, verbose
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save_model('best_model')
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Save final model
        self.save_model('final_model')

        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        max_grad_norm: float,
        verbose: bool
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        iterator = tqdm(train_loader, desc="Training", disable=not verbose)

        for batch in iterator:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if 'token_type_ids' in batch:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if 'token_type_ids' in batch:
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=batch['token_type_ids'].to(self.device)
                    )
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test data."""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if 'token_type_ids' in batch:
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=batch['token_type_ids'].to(self.device)
                    )
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        results = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs)
        }

        # Save results
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def save_model(self, name: str = 'model'):
        """Save model, tokenizer, and config."""
        save_path = self.output_dir / name
        save_path.mkdir(exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), save_path / 'pytorch_model.bin')

        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        print(f"Model saved to: {save_path}")

    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(Path(path) / 'pytorch_model.bin', map_location=self.device)
        )
        self.model.to(self.device)
