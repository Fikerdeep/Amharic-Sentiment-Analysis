# Amharic Sentiment Analysis

A deep learning-based sentiment analysis system for Amharic (Ethiopian) text. This project implements multiple neural network architectures for binary sentiment classification (positive/negative).

## Features

- **Multiple Model Architectures**: CNN, BiLSTM, GRU, and hybrid CNN-BiLSTM
- **Framework Support**: TensorFlow/Keras, PyTorch, and Hugging Face Transformers
- **Amharic-Specific Preprocessing**: Handles Ge'ez script character variants and labialized characters
- **Production-Ready API**: FastAPI REST API with Docker support
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
│   ├── models/                  # TensorFlow/Keras models
│   ├── pytorch/                 # PyTorch models
│   ├── transformers/            # Hugging Face Transformers
│   ├── training/                # Training pipeline
│   ├── evaluation/              # Metrics and visualization
│   └── utils/                   # Configuration and logging
├── api/                         # FastAPI REST API
├── docker/                      # Docker configuration
├── configs/                     # YAML configuration files
├── scripts/                     # Utility scripts
├── notebooks/                   # Jupyter notebooks
└── dataset/                     # Training data
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

# Make predictions
amharic-sentiment predict --model saved_models/model.keras \
    --tokenizer saved_models/tokenizer.pkl \
    --text "ጥሩ ስራ ነው"
```

---

## REST API

### Running the API

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Run locally
python scripts/run_api.py --port 8000

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch prediction (up to 100) |
| `/model/load` | POST | Load a different model |
| `/model/info` | GET | Current model information |
| `/docs` | GET | Swagger documentation |

### API Usage Examples

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "ጥሩ ስራ ነው ተባረኩ"}'
```

Response:
```json
{
  "success": true,
  "result": {
    "text": "ጥሩ ስራ ነው ተባረኩ",
    "sentiment": "positive",
    "confidence": 0.92,
    "probability": 0.92
  },
  "model_type": "tensorflow",
  "processing_time_ms": 45.2
}
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ጥሩ ስራ ነው", "መጥፎ ነገር"]}'
```

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t amharic-sentiment-api .

# Run the container
docker run -p 8000:8000 \
  -v ./saved_models:/app/saved_models \
  amharic-sentiment-api
```

### Using Docker Compose

```bash
cd docker

# Start the API
docker-compose up -d

# Start with Nginx (production)
docker-compose --profile production up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | tensorflow | Model type (tensorflow/pytorch/transformer) |
| `MODEL_PATH` | /app/saved_models/model.keras | Path to model file |
| `TOKENIZER_PATH` | /app/saved_models/tokenizer.pkl | Path to tokenizer |
| `MAX_LEN` | 20 | Maximum sequence length |

---

## PyTorch Models

```python
from amharic_sentiment.pytorch.models import CNNBiLSTMClassifier
from amharic_sentiment.pytorch.training import PyTorchTrainer

# Create model
model = CNNBiLSTMClassifier(vocab_size=15000, embedding_dim=100)

# Train
trainer = PyTorchTrainer(model, device='cuda')
trainer.compile(optimizer='adam', learning_rate=0.001)
trainer.fit(train_loader, val_loader, epochs=10)
```

## Transformer Models

```python
from amharic_sentiment.transformers import TransformerClassifier, TransformerTrainer

# Load XLM-RoBERTa for Amharic
model = TransformerClassifier('xlm-roberta', num_labels=1)

# Fine-tune
trainer = TransformerTrainer(model)
trainer.train(train_loader, val_loader, epochs=5, learning_rate=2e-5)
```

Available models:
- `mbert` - Multilingual BERT
- `xlm-roberta` - XLM-RoBERTa Base
- `xlm-roberta-large` - XLM-RoBERTa Large
- `afro-xlmr` - African languages focused XLM-R

---

## Preprocessing Pipeline

```python
from amharic_sentiment.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
clean_text = pipeline.process("ሰላም!! https://example.com ይህ ጥሩ ነው 123")
# Output: "ሰላም ይህ ጥሩ ነው"
```

Features:
- URL removal
- Amharic punctuation cleaning (፤።፡፣ etc.)
- Character variant normalization (ሃ→ሀ, ጸ→ፀ)
- Labialized character normalization (ሉዋ→ሏ)
- Emoji and special character removal

---

## Configuration

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

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/train.py` | Train models |
| `scripts/run_api.py` | Run the REST API |
| `scripts/export_model.py` | Export model for production |
| `scripts/test_api.py` | Test API endpoints |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black amharic_sentiment/
isort amharic_sentiment/
```

---

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{amharic_sentiment_analysis,
  author = {Fikerte Shalemo},
  title = {Amharic Sentiment Analysis},
  year = {2024},
  url = {https://github.com/Fikerdeep/Amharic-Sentiment-Analysis}
}
```
