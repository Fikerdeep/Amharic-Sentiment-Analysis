#!/usr/bin/env python
"""
Training script for Amharic sentiment analysis models.

This script provides a simple way to train and evaluate models
with customizable parameters.

Usage:
    python scripts/train.py --model cnn_bilstm --epochs 10
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from amharic_sentiment.data.dataset import AmharicSentimentDataset
from amharic_sentiment.training.trainer import Trainer
from amharic_sentiment.utils.config import Config, load_config
from amharic_sentiment.utils.logger import TrainingLogger


def main():
    parser = argparse.ArgumentParser(
        description="Train Amharic sentiment analysis model"
    )

    # Data arguments
    parser.add_argument(
        "--positive",
        type=str,
        default="dataset/postive comment.txt",
        help="Path to positive samples file"
    )
    parser.add_argument(
        "--negative",
        type=str,
        default="dataset/negative comment.txt",
        help="Path to negative samples file"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_bilstm",
        choices=["cnn", "bilstm", "gru", "cnn_bilstm"],
        help="Model architecture"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights for imbalanced data"
    )

    # Data processing arguments
    parser.add_argument(
        "--max-words",
        type=int,
        default=15000,
        help="Maximum vocabulary size"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=20,
        help="Maximum sequence length"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="experiments",
        help="Output directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file"
    )

    args = parser.parse_args()

    # Setup logger
    logger = TrainingLogger()
    logger.info("Starting Amharic Sentiment Analysis Training")

    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = Config()

    # Override with CLI arguments
    config.data.positive_file = args.positive
    config.data.negative_file = args.negative
    config.data.max_words = args.max_words
    config.data.max_len = args.max_len
    config.model.model_type = args.model
    config.model.embedding_dim = args.embedding_dim
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.use_class_weights = args.use_class_weights
    config.output_dir = args.output

    # Log configuration
    logger.log_config(config.to_dict())

    # Load dataset
    logger.info("Loading dataset...")
    dataset = AmharicSentimentDataset(
        max_words=config.data.max_words,
        max_len=config.data.max_len
    )

    try:
        dataset.load_from_files(
            config.data.positive_file,
            config.data.negative_file
        )
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)

    stats = dataset.get_stats()
    logger.info(f"Dataset loaded: {stats['total_samples']} samples")
    logger.info(f"  Positive: {stats['positive_samples']}")
    logger.info(f"  Negative: {stats['negative_samples']}")

    # Create trainer
    model_params = {
        'embedding_dim': config.model.embedding_dim
    }

    trainer = Trainer(
        model_type=config.model.model_type,
        dataset=dataset,
        output_dir=config.output_dir,
        experiment_name=args.name,
        model_params=model_params
    )

    # Train model
    logger.info(f"Training {config.model.model_type} model...")
    history = trainer.train(
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        use_class_weights=config.training.use_class_weights
    )

    # Evaluate
    logger.info("Evaluating model...")
    results = trainer.evaluate()
    logger.log_evaluation(results)

    # Save model
    model_path = trainer.save_model()
    logger.info(f"Model saved to: {model_path}")

    # Save tokenizer
    tokenizer_path = Path(trainer.output_dir) / "tokenizer.pkl"
    dataset.save_tokenizer(str(tokenizer_path))
    logger.info(f"Tokenizer saved to: {tokenizer_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
