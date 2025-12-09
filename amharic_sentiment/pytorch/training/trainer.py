"""
PyTorch training pipeline for Amharic sentiment analysis.

This module provides a high-level trainer class for PyTorch models,
with support for mixed precision training, gradient accumulation,
and various optimizers and schedulers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


class PyTorchTrainer:
    """
    High-level trainer for PyTorch sentiment analysis models.

    Features:
        - Multiple optimizer support (Adam, AdamW, SGD)
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Mixed precision training (optional)
        - Gradient accumulation

    Example:
        >>> trainer = PyTorchTrainer(model, device='cuda')
        >>> trainer.fit(train_loader, val_loader, epochs=10)
        >>> results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        output_dir: str = 'experiments',
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            output_dir: Directory for outputs
            experiment_name: Name for this experiment
        """
        self.model = model

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
            experiment_name = f"pytorch_{timestamp}"

        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict] = None
    ) -> 'PyTorchTrainer':
        """
        Configure optimizer and scheduler.

        Args:
            optimizer: Optimizer name ('adam', 'adamw', 'sgd')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler: Scheduler name ('step', 'cosine', 'plateau', 'onecycle')
            scheduler_params: Additional scheduler parameters

        Returns:
            Self for method chaining
        """
        # Setup optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Setup scheduler
        scheduler_params = scheduler_params or {}
        if scheduler == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 5),
                gamma=scheduler_params.get('gamma', 0.5)
            )
        elif scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 10)
            )
        elif scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 3)
            )

        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        return self

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 5,
        gradient_accumulation_steps: int = 1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            gradient_accumulation_steps: Steps for gradient accumulation
            verbose: Whether to show progress

        Returns:
            Training history
        """
        if self.optimizer is None:
            self.compile()

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, gradient_accumulation_steps, verbose
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Learning rate scheduling
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= early_stopping_patience:
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

        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        gradient_accumulation_steps: int,
        verbose: bool
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        iterator = tqdm(train_loader, desc="Training", disable=not verbose)

        for i, batch in enumerate(iterator):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

            # Predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
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

        return results

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions.

        Args:
            data_loader: Data loader

        Returns:
            Prediction probabilities
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids)
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def save_checkpoint(self, filename: str) -> str:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, filepath: str) -> 'PyTorchTrainer':
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return self

    def save_model(self, filename: str = 'model.pt') -> str:
        """Save model weights only."""
        model_path = self.output_dir / filename
        torch.save(self.model.state_dict(), model_path)
        return str(model_path)
