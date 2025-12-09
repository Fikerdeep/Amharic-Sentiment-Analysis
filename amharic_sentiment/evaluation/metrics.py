"""
Evaluation metrics for sentiment analysis models.

This module provides functions for evaluating model performance,
including accuracy, precision, recall, F1-score, and detailed
classification reports.
"""

from typing import Dict, Any, Optional, Union
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a model on test data.

    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold

    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba > threshold).astype(int)

    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'threshold': threshold
    }

    return results


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
    output_dict: bool = True
) -> Union[str, Dict[str, Any]]:
    """
    Generate a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for target classes
        output_dict: Whether to return as dictionary

    Returns:
        Classification report as string or dictionary
    """
    if target_names is None:
        target_names = ['Negative', 'Positive']

    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate basic classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> tuple:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    metric_funcs = {
        'f1': f1_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}")

    metric_func = metric_funcs[metric]

    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        score = metric_func(y_true, y_pred, zero_division=0) if metric != 'accuracy' else metric_func(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def get_roc_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Get ROC curve data for plotting.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities

    Returns:
        Dictionary with fpr, tpr, thresholds, and auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc
    }


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test set.

    Args:
        models: Dictionary mapping model names to model objects
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba > 0.5).astype(int)

        results[name] = calculate_metrics(y_test, y_pred, y_proba)

    return results
