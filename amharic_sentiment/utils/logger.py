"""
Logging utilities for Amharic sentiment analysis.

This module provides standardized logging configuration for the package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "amharic_sentiment",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file name
        log_dir: Directory for log files
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "amharic_sentiment") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If no handlers, set up default
    if not logger.handlers:
        return setup_logger(name)

    return logger


class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides convenient methods for logging training metrics
    and progress.
    """

    def __init__(self, name: str = "training", log_dir: str = "logs"):
        """
        Initialize the training logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{timestamp}.log"

        self.logger = setup_logger(
            name=name,
            log_file=log_file,
            log_dir=log_dir
        )

    def log_config(self, config: dict) -> None:
        """Log configuration."""
        self.logger.info("=" * 50)
        self.logger.info("Training Configuration")
        self.logger.info("=" * 50)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch metrics."""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")

    def log_evaluation(self, results: dict) -> None:
        """Log evaluation results."""
        self.logger.info("=" * 50)
        self.logger.info("Evaluation Results")
        self.logger.info("=" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
