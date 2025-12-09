"""Evaluation metrics and utilities."""

from amharic_sentiment.evaluation.metrics import evaluate_model, get_classification_report
from amharic_sentiment.evaluation.visualize import plot_training_history, plot_confusion_matrix

__all__ = ["evaluate_model", "get_classification_report", "plot_training_history", "plot_confusion_matrix"]
