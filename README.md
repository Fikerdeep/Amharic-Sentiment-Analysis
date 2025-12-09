# Amharic Sentiment Analysis

A deep learning-based sentiment analysis system for Amharic (Ethiopian) text. This project implements multiple neural network architectures for binary sentiment classification (positive/negative).

## Features

- **Multiple Model Architectures**: CNN, BiLSTM, GRU, and hybrid CNN-BiLSTM
- **Amharic-Specific Preprocessing**: Handles Ge'ez script character variants and labialized characters
- **Modular Design**: Clean, reusable code organized as a Python package
- **Easy to Use**: Simple API and CLI for training and prediction
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC metrics

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CNN | 84.8% | 80.4% | 73.7% | - |
| GRU | 88.6% | 88.0% | 91.5% | - |
| BiLSTM | 87.6% | 84.2% | 92.9% | - |
| **CNN-BiLSTM** | **91.6%** | **90.5%** | **93.9%** | - |

## Project Structure

```
Amharic-Sentiment-Analysis/
├── amharic_sentiment/           # Main package
│   ├── preprocessing/           # Text cleaning and normalization
│   ├── data/                    # Dataset and data loading utilities
│   ├── models/                  # Neural network architectures
│   ├── training/                # Training pipeline and callbacks
│   ├── evaluation/              # Metrics and visualization
│   └── utils/                   # Configuration and logging
├── configs/                     # YAML configuration files
├── scripts/                     # Training and utility scripts
├── notebooks/                   # Jupyter notebooks with examples
├── dataset/                     # Training data
├── experiments/                 # Saved experiments and models
└── logs/                        # Training logs
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Fikerdeep/Amharic-Sentiment-Analysis.git
cd Amharic-Sentiment-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Using Python API

```python
from amharic_sentiment.data.dataset import AmharicSentimentDataset
from amharic_sentiment.training.trainer import Trainer

# Load data
dataset = AmharicSentimentDataset(max_words=15000, max_len=20)
dataset.load_from_files(
    positive_file='dataset/postive comment.txt',
    negative_file='dataset/negative comment.txt'
)

# Train model
trainer = Trainer(
    model_type='cnn_bilstm',
    dataset=dataset,
    output_dir='experiments'
)
trainer.train(epochs=10, batch_size=32)

# Evaluate
results = trainer.evaluate()
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Using CLI

```bash
# Train a model
python scripts/train.py --model cnn_bilstm --epochs 10

# Or use the installed CLI
amharic-sentiment train --model cnn_bilstm --epochs 10

# Make predictions
amharic-sentiment predict --model saved_models/model.keras \
    --tokenizer saved_models/tokenizer.pkl \
    --text "ጥሩ ስራ ነው"
```

### Using Configuration Files

```bash
# Train with config file
python scripts/train.py --config configs/cnn_bilstm.yaml
```

## Model Architectures

### 1. CNN (Convolutional Neural Network)
- Embedding → Conv1D → GlobalMaxPooling → Dense → Output
- Good for capturing local n-gram patterns

### 2. BiLSTM (Bidirectional LSTM)
- Embedding → BiLSTM → BiLSTM → Dense → Output
- Captures long-range dependencies in both directions

### 3. GRU (Gated Recurrent Unit)
- Embedding → GRU → GRU → Output
- Efficient alternative to LSTM

### 4. CNN-BiLSTM (Hybrid)
- Embedding → Conv1D → MaxPool → BiLSTM → Output
- Combines local feature extraction with sequence modeling
- **Best performing model**

## Preprocessing Pipeline

The preprocessing pipeline handles Amharic-specific text cleaning:

```python
from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
clean_text = pipeline.process("ሰላም!! https://example.com ይህ ጥሩ ነው 123")
# Output: "ሰላም ይህ ጥሩ ነው"
```

### Features:
- URL removal
- Amharic punctuation cleaning (፤።፡፣ etc.)
- Special character removal
- English text and number removal
- Emoji removal
- Character variant normalization (ሃ→ሀ, ጸ→ፀ, etc.)
- Labialized character normalization (ሉዋ→ሏ, etc.)

## Configuration

Configuration can be done via YAML files:

```yaml
# configs/default.yaml
data:
  positive_file: "dataset/postive comment.txt"
  negative_file: "dataset/negative comment.txt"
  max_words: 15000
  max_len: 20

model:
  model_type: "cnn_bilstm"
  embedding_dim: 32
  filters: 64
  lstm_units: 64

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
```

## Dataset

The dataset contains ~5,000 labeled Amharic comments:
- **Positive samples**: 2,721
- **Negative samples**: 2,704

Sample data format (one comment per line):
```
ታዛዥ ነን የምንችለውን ሁሉ ለማደረግ እግዚአብሔር ብቻ ከናንተ ጋር ይሁን
ድል ከእውነተኛው ተባዳዮች ጋር ናት ጀግናዬ ነህ
```

## Notebooks

See the `notebooks/` directory for interactive examples:
- `01_getting_started.ipynb`: Complete tutorial on using the package

## API Reference

### AmharicSentimentDataset

```python
dataset = AmharicSentimentDataset(max_words=15000, max_len=20)
dataset.load_from_files(positive_file, negative_file)
dataset.preprocess()
dataset.fit_tokenizer()
X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_data()
```

### Trainer

```python
trainer = Trainer(model_type='cnn_bilstm', dataset=dataset)
trainer.train(epochs=10, batch_size=32)
results = trainer.evaluate()
trainer.save_model()
```

### Models

```python
from amharic_sentiment.models import CNN, BiLSTM, GRU, CNNBiLSTM

model = CNNBiLSTM(vocab_size=10000, embedding_dim=32, max_len=20)
model.build()
model.compile()
model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
flake8 src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{amharic_sentiment_analysis,
  author = {Fikerte Shalemo},
  title = {Amharic Sentiment Analysis},
  year = {2024},
  url = {https://github.com/Fikerdeep/Amharic-Sentiment-Analysis}
}
```

## Acknowledgments

- Dataset collected from various Ethiopian social media platforms
- Inspired by research on low-resource language NLP
