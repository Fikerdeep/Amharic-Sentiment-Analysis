"""
Visualization utilities for model evaluation.

This module provides functions for creating plots and visualizations
of model training and evaluation results.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tensorflow.keras.callbacks import History


def plot_training_history(
    history: History,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
) -> None:
    """
    Plot training history metrics.

    Args:
        history: Keras training history object
        metrics: List of metrics to plot (default: loss, accuracy)
        save_path: Path to save the figure
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']

    history_dict = history.history if hasattr(history, 'history') else history

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history_dict:
            ax.plot(history_dict[metric], label=f'Train {metric}')

            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                ax.plot(history_dict[val_metric], label=f'Val {metric}')

            ax.set_title(f'Model {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion_mat: Confusion matrix array
        labels: Class labels
        save_path: Path to save the figure
        figsize: Figure size
        cmap: Colormap name
    """
    if labels is None:
        labels = ['Negative', 'Positive']

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
        ax=ax
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve score
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot comparison of metrics across multiple models.

    Args:
        results: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to compare
        save_path: Path to save the figure
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_prediction_distribution(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
) -> None:
    """
    Plot distribution of prediction probabilities.

    Args:
        y_proba: Prediction probabilities
        y_true: True labels
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distribution by class
    ax = axes[0]
    for label, name in [(0, 'Negative'), (1, 'Positive')]:
        mask = y_true == label
        ax.hist(y_proba[mask], bins=30, alpha=0.7, label=name)

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overall distribution
    ax = axes[1]
    ax.hist(y_proba, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Overall Prediction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def save_all_plots(
    history: History,
    confusion_mat: np.ndarray,
    roc_data: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Save all evaluation plots to a directory.

    Args:
        history: Training history
        confusion_mat: Confusion matrix
        roc_data: ROC curve data dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training history
    plot_training_history(
        history,
        save_path=str(output_path / 'training_history.png')
    )

    # Confusion matrix
    plot_confusion_matrix(
        confusion_mat,
        save_path=str(output_path / 'confusion_matrix.png')
    )

    # ROC curve
    plot_roc_curve(
        roc_data['fpr'],
        roc_data['tpr'],
        roc_data['auc'],
        save_path=str(output_path / 'roc_curve.png')
    )
