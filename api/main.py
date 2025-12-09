"""
FastAPI application for Amharic Sentiment Analysis.

Provides REST API endpoints for sentiment prediction.
"""

import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    TextInput,
    BatchTextInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    SentimentResult,
    SentimentLabel
)
from api.model_service import model_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_TYPE = os.getenv("MODEL_TYPE", "tensorflow")
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/model.keras")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "saved_models/tokenizer.pkl")
MAX_LEN = int(os.getenv("MAX_LEN", "20"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Amharic Sentiment Analysis API...")

    # Load model on startup if paths are configured
    if os.path.exists(MODEL_PATH):
        try:
            success = model_manager.load_model(
                model_type=MODEL_TYPE,
                model_path=MODEL_PATH,
                tokenizer_path=TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else None,
                max_len=MAX_LEN
            )
            if success:
                logger.info(f"Model loaded successfully: {MODEL_TYPE}")
            else:
                logger.warning("Failed to load model on startup")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    else:
        logger.warning(f"Model path not found: {MODEL_PATH}")

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Amharic Sentiment Analysis API",
    description="""
    REST API for analyzing sentiment in Amharic (Ethiopian) text.

    ## Features
    - Single text prediction
    - Batch prediction (up to 100 texts)
    - Multiple model support (TensorFlow, PyTorch, Transformers)

    ## Sentiment Labels
    - **positive**: Text expresses positive sentiment
    - **negative**: Text expresses negative sentiment

    ## Models
    - **tensorflow**: Keras CNN-BiLSTM model
    - **pytorch**: PyTorch CNN-BiLSTM model
    - **transformer**: XLM-RoBERTa fine-tuned model
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Amharic Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health status.

    Returns service status and model information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_manager.is_loaded(),
        model_type=model_manager.get_current_type()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prediction"]
)
async def predict_single(input_data: TextInput):
    """
    Predict sentiment for a single text.

    - **text**: Amharic text to analyze (required)

    Returns sentiment prediction with confidence score.
    """
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load a model first."
        )

    try:
        start_time = time.time()

        # Get prediction
        results = model_manager.predict([input_data.text])
        sentiment, probability = results[0]

        # Calculate confidence
        confidence = probability if probability > 0.5 else 1 - probability

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            success=True,
            result=SentimentResult(
                text=input_data.text,
                sentiment=SentimentLabel(sentiment),
                confidence=round(confidence, 4),
                probability=round(probability, 4)
            ),
            model_type=model_manager.get_current_type(),
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prediction"]
)
async def predict_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts.

    - **texts**: List of Amharic texts to analyze (max 100)

    Returns list of sentiment predictions.
    """
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load a model first."
        )

    if len(input_data.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 texts per batch"
        )

    try:
        start_time = time.time()

        # Get predictions
        predictions = model_manager.predict(input_data.texts)

        # Format results
        results = []
        for text, (sentiment, probability) in zip(input_data.texts, predictions):
            confidence = probability if probability > 0.5 else 1 - probability
            results.append(SentimentResult(
                text=text,
                sentiment=SentimentLabel(sentiment),
                confidence=round(confidence, 4),
                probability=round(probability, 4)
            ))

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            success=True,
            results=results,
            model_type=model_manager.get_current_type(),
            total_count=len(results),
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load", tags=["Model Management"])
async def load_model(
    model_type: str = Query(..., description="Model type: tensorflow, pytorch, transformer"),
    model_path: str = Query(..., description="Path to model file"),
    tokenizer_path: Optional[str] = Query(None, description="Path to tokenizer file"),
    max_len: int = Query(20, description="Maximum sequence length")
):
    """
    Load a model for inference.

    This endpoint allows switching between different model types.
    """
    valid_types = ["tensorflow", "pytorch", "transformer"]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of: {valid_types}"
        )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Model path not found: {model_path}"
        )

    try:
        success = model_manager.load_model(
            model_type=model_type,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_len=max_len
        )

        if success:
            return {
                "success": True,
                "message": f"Model loaded successfully",
                "model_type": model_type
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )

    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model Management"])
async def model_info():
    """Get information about the currently loaded model."""
    return {
        "loaded": model_manager.is_loaded(),
        "model_type": model_manager.get_current_type(),
        "available_types": ["tensorflow", "pytorch", "transformer"]
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
