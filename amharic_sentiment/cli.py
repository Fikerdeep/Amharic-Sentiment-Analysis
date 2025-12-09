"""
Command-line interface for Amharic sentiment analysis.

This module provides a CLI for training, evaluating, and predicting
with sentiment analysis models.
"""

import argparse
import sys
from pathlib import Path


def train_command(args):
    """Handle the train command."""
    from amharic_sentiment.data.dataset import AmharicSentimentDataset
    from amharic_sentiment.training.trainer import Trainer
    from amharic_sentiment.utils.config import Config

    print("=" * 60)
    print("Amharic Sentiment Analysis - Training")
    print("=" * 60)

    # Load config if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override config with CLI arguments
    if args.model:
        config.model.model_type = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    # Load dataset
    print(f"\nLoading data...")
    dataset = AmharicSentimentDataset(
        max_words=config.data.max_words,
        max_len=config.data.max_len
    )

    positive_file = args.positive or config.data.positive_file
    negative_file = args.negative or config.data.negative_file

    dataset.load_from_files(positive_file, negative_file)
    print(f"  Loaded {len(dataset.texts)} samples")

    # Create trainer
    trainer = Trainer(
        model_type=config.model.model_type,
        dataset=dataset,
        output_dir=args.output or config.output_dir,
        experiment_name=args.name
    )

    # Train
    print(f"\nTraining {config.model.model_type} model...")
    trainer.train(
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        use_class_weights=config.training.use_class_weights
    )

    # Evaluate
    print("\nEvaluating model...")
    results = trainer.evaluate()

    # Save model
    model_path = trainer.save_model()
    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


def predict_command(args):
    """Handle the predict command."""
    from amharic_sentiment.models.base import BaseModel
    from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pickle

    print("=" * 60)
    print("Amharic Sentiment Analysis - Prediction")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    # Preprocessing pipeline
    pipeline = PreprocessingPipeline()

    # Get text input
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Enter text (press Ctrl+D to finish):")
        texts = []
        try:
            while True:
                line = input()
                if line.strip():
                    texts.append(line.strip())
        except EOFError:
            pass

    if not texts:
        print("No text provided!")
        return

    # Process and predict
    print(f"\nProcessing {len(texts)} text(s)...")

    for text in texts:
        # Preprocess
        processed = pipeline.process(text)

        # Tokenize
        sequences = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequences, maxlen=args.max_len, padding='post')

        # Predict
        prob = model.predict(padded, verbose=0)[0][0]
        sentiment = "Positive" if prob > 0.5 else "Negative"
        confidence = prob if prob > 0.5 else 1 - prob

        print(f"\nText: {text[:50]}...")
        print(f"  Sentiment: {sentiment}")
        print(f"  Confidence: {confidence:.2%}")


def evaluate_command(args):
    """Handle the evaluate command."""
    from amharic_sentiment.data.dataset import AmharicSentimentDataset
    from amharic_sentiment.evaluation.metrics import evaluate_model, get_classification_report
    from amharic_sentiment.evaluation.visualize import (
        plot_confusion_matrix, plot_training_history
    )
    from tensorflow.keras.models import load_model
    import pickle
    import numpy as np

    print("=" * 60)
    print("Amharic Sentiment Analysis - Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load test data
    print(f"Loading test data...")
    dataset = AmharicSentimentDataset(max_words=15000, max_len=args.max_len)
    dataset.load_from_files(args.positive, args.negative)
    dataset.preprocess()
    dataset.tokenizer = tokenizer
    dataset.vocab_size = len(tokenizer.word_index) + 1

    X = dataset.tokenize()
    y = dataset.labels

    # Evaluate
    print("\nEvaluating...")

    # Wrap model in a simple class for compatibility
    class ModelWrapper:
        def __init__(self, keras_model):
            self.model = keras_model

        def predict_proba(self, X):
            return self.model.predict(X, verbose=0).flatten()

    wrapper = ModelWrapper(model)
    results = evaluate_model(wrapper, X, y)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print(f"  ROC AUC:   {results['roc_auc']:.4f}")

    # Classification report
    y_pred = (wrapper.predict_proba(X) > 0.5).astype(int)
    report = get_classification_report(y, y_pred, output_dict=False)
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix if requested
    if args.plot:
        plot_confusion_matrix(results['confusion_matrix'])


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Amharic Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  amharic-sentiment train --positive data/positive.txt --negative data/negative.txt

  # Train with specific model type
  amharic-sentiment train --model cnn_bilstm --epochs 20

  # Make predictions
  amharic-sentiment predict --model saved_models/model.keras --tokenizer saved_models/tokenizer.pkl --text "ጥሩ ነው"

  # Evaluate a model
  amharic-sentiment evaluate --model saved_models/model.keras --tokenizer saved_models/tokenizer.pkl
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a sentiment analysis model")
    train_parser.add_argument("--positive", type=str, help="Path to positive samples file")
    train_parser.add_argument("--negative", type=str, help="Path to negative samples file")
    train_parser.add_argument("--model", type=str, choices=["cnn", "bilstm", "gru", "cnn_bilstm"],
                              help="Model architecture")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--output", type=str, help="Output directory")
    train_parser.add_argument("--name", type=str, help="Experiment name")
    train_parser.add_argument("--config", type=str, help="Path to config YAML file")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    predict_parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    predict_parser.add_argument("--text", type=str, help="Text to classify")
    predict_parser.add_argument("--file", type=str, help="File with texts to classify")
    predict_parser.add_argument("--max-len", type=int, default=20, help="Maximum sequence length")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    eval_parser.add_argument("--positive", type=str, required=True, help="Path to positive test data")
    eval_parser.add_argument("--negative", type=str, required=True, help="Path to negative test data")
    eval_parser.add_argument("--max-len", type=int, default=20, help="Maximum sequence length")
    eval_parser.add_argument("--plot", action="store_true", help="Show plots")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
