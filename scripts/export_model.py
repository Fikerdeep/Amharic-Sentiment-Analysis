#!/usr/bin/env python
"""
Script to export trained models for production deployment.

Exports model and tokenizer to a format ready for API serving.

Usage:
    python scripts/export_model.py --experiment experiments/cnn_bilstm_demo --output saved_models
"""

import argparse
import shutil
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_tensorflow_model(experiment_dir: Path, output_dir: Path):
    """Export TensorFlow/Keras model."""
    print("Exporting TensorFlow model...")

    # Find model file
    model_files = list(experiment_dir.glob("**/*.keras")) + list(experiment_dir.glob("**/*.h5"))
    if not model_files:
        raise FileNotFoundError("No Keras model found in experiment directory")

    model_path = model_files[0]
    print(f"  Found model: {model_path}")

    # Copy model
    output_model = output_dir / "model.keras"
    shutil.copy(model_path, output_model)
    print(f"  Copied to: {output_model}")

    # Find and copy tokenizer
    tokenizer_files = list(experiment_dir.glob("**/tokenizer.pkl"))
    if tokenizer_files:
        tokenizer_path = tokenizer_files[0]
        output_tokenizer = output_dir / "tokenizer.pkl"
        shutil.copy(tokenizer_path, output_tokenizer)
        print(f"  Tokenizer copied to: {output_tokenizer}")

    # Copy config if exists
    config_files = list(experiment_dir.glob("**/config.json"))
    if config_files:
        shutil.copy(config_files[0], output_dir / "config.json")

    print("TensorFlow model exported successfully!")


def export_pytorch_model(experiment_dir: Path, output_dir: Path):
    """Export PyTorch model."""
    print("Exporting PyTorch model...")

    # Find model file
    model_files = list(experiment_dir.glob("**/*.pt"))
    if not model_files:
        raise FileNotFoundError("No PyTorch model found in experiment directory")

    model_path = model_files[0]
    print(f"  Found model: {model_path}")

    # Copy model
    output_model = output_dir / "model.pt"
    shutil.copy(model_path, output_model)
    print(f"  Copied to: {output_model}")

    # Find and copy tokenizer
    tokenizer_files = list(experiment_dir.glob("**/tokenizer.pkl"))
    if tokenizer_files:
        output_tokenizer = output_dir / "tokenizer.pkl"
        shutil.copy(tokenizer_files[0], output_tokenizer)
        print(f"  Tokenizer copied to: {output_tokenizer}")

    print("PyTorch model exported successfully!")


def export_transformer_model(experiment_dir: Path, output_dir: Path):
    """Export Transformer model."""
    print("Exporting Transformer model...")

    # Find best model directory
    best_model_dirs = list(experiment_dir.glob("**/best_model"))
    if not best_model_dirs:
        # Look for pytorch_model.bin
        model_files = list(experiment_dir.glob("**/pytorch_model.bin"))
        if model_files:
            model_dir = model_files[0].parent
        else:
            raise FileNotFoundError("No Transformer model found")
    else:
        model_dir = best_model_dirs[0]

    print(f"  Found model directory: {model_dir}")

    # Copy entire model directory
    output_model_dir = output_dir / "transformer_model"
    if output_model_dir.exists():
        shutil.rmtree(output_model_dir)
    shutil.copytree(model_dir, output_model_dir)
    print(f"  Copied to: {output_model_dir}")

    print("Transformer model exported successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model for production"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="saved_models",
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="tensorflow",
        choices=["tensorflow", "pytorch", "transformer"],
        help="Type of model to export"
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment)
    output_dir = Path(args.output)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export based on model type
    if args.model_type == "tensorflow":
        export_tensorflow_model(experiment_dir, output_dir)
    elif args.model_type == "pytorch":
        export_pytorch_model(experiment_dir, output_dir)
    elif args.model_type == "transformer":
        export_transformer_model(experiment_dir, output_dir)

    # Create metadata file
    metadata = {
        "model_type": args.model_type,
        "source_experiment": str(experiment_dir),
        "exported_files": [f.name for f in output_dir.iterdir()]
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExport complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
