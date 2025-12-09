#!/usr/bin/env python
"""
Script to run the Amharic Sentiment Analysis API locally.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --model-path saved_models/model.keras --port 8000
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run Amharic Sentiment Analysis API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="tensorflow",
        choices=["tensorflow", "pytorch", "transformer"],
        help="Model type to load"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/model.keras",
        help="Path to model file"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="saved_models/tokenizer.pkl",
        help="Path to tokenizer file"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers"
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["MODEL_TYPE"] = args.model_type
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["TOKENIZER_PATH"] = args.tokenizer_path

    # Run server
    import uvicorn

    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║     Amharic Sentiment Analysis API                   ║
    ╠══════════════════════════════════════════════════════╣
    ║  Host:       {args.host:<40} ║
    ║  Port:       {args.port:<40} ║
    ║  Model:      {args.model_type:<40} ║
    ║  Docs:       http://{args.host}:{args.port}/docs{' '*21} ║
    ╚══════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
