"""
Keras callbacks for training sentiment analysis models.

This module provides pre-configured callbacks for model training,
including early stopping, model checkpointing, and learning rate scheduling.
"""

from typing import List, Optional
from pathlib import Path

from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)


def get_callbacks(
    model_name: str = "model",
    checkpoint_dir: str = "saved_models",
    log_dir: str = "logs",
    monitor: str = "val_loss",
    patience: int = 5,
    min_delta: float = 0.001,
    enable_early_stopping: bool = True,
    enable_checkpoint: bool = True,
    enable_lr_scheduler: bool = True,
    enable_tensorboard: bool = True,
    enable_csv_logger: bool = True
) -> List[Callback]:
    """
    Get a list of training callbacks.

    Args:
        model_name: Name for saving model checkpoints
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for logs
        monitor: Metric to monitor for callbacks
        patience: Patience for early stopping
        min_delta: Minimum change to qualify as improvement
        enable_early_stopping: Enable early stopping callback
        enable_checkpoint: Enable model checkpoint callback
        enable_lr_scheduler: Enable learning rate reduction callback
        enable_tensorboard: Enable TensorBoard callback
        enable_csv_logger: Enable CSV logging callback

    Returns:
        List of configured Keras callbacks
    """
    callbacks = []

    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Early stopping
    if enable_early_stopping:
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

    # Model checkpoint
    if enable_checkpoint:
        checkpoint_path = Path(checkpoint_dir) / f"{model_name}_best.keras"
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)

    # Learning rate scheduler
    if enable_lr_scheduler:
        lr_scheduler = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)

    # TensorBoard
    if enable_tensorboard:
        tensorboard_dir = Path(log_dir) / "tensorboard" / model_name
        tensorboard = TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)

    # CSV Logger
    if enable_csv_logger:
        csv_path = Path(log_dir) / f"{model_name}_training.csv"
        csv_logger = CSVLogger(
            filename=str(csv_path),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)

    return callbacks


class TrainingProgressCallback(Callback):
    """
    Custom callback for tracking training progress.

    Provides detailed logging of training metrics and can be extended
    for custom progress tracking needs.
    """

    def __init__(self, print_freq: int = 1):
        """
        Initialize the callback.

        Args:
            print_freq: Frequency of progress printing (epochs)
        """
        super().__init__()
        self.print_freq = print_freq
        self.epoch_logs = []

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        self.epoch_logs.append(logs.copy())

        if (epoch + 1) % self.print_freq == 0:
            metrics_str = " - ".join([
                f"{k}: {v:.4f}" for k, v in logs.items()
            ])
            print(f"Epoch {epoch + 1}: {metrics_str}")

    def on_train_end(self, logs: Optional[dict] = None):
        """Called at the end of training."""
        if self.epoch_logs:
            best_epoch = min(
                range(len(self.epoch_logs)),
                key=lambda i: self.epoch_logs[i].get('val_loss', float('inf'))
            )
            print(f"\nBest epoch: {best_epoch + 1}")
            print(f"Best val_loss: {self.epoch_logs[best_epoch].get('val_loss', 'N/A'):.4f}")
