#!/usr/bin/env python
"""
Script to test the Amharic Sentiment Analysis API.

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --url http://localhost:8000
"""

import argparse
import requests
import json
from typing import List


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_single_prediction(base_url: str) -> bool:
    """Test single prediction endpoint."""
    print("\n2. Testing /predict endpoint (single text)...")

    test_cases = [
        ("ጥሩ ስራ ነው ተባረኩ", "positive"),  # Good work, be blessed
        ("በጣም መጥፎ ውሳኔ ነው", "negative"),  # Very bad decision
        ("እግዚአብሔር ይስጥልን", "positive"),  # May God give us
    ]

    success = True
    for text, expected in test_cases:
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": text}
            )
            data = response.json()

            if response.status_code == 200:
                result = data.get("result", {})
                sentiment = result.get("sentiment", "unknown")
                confidence = result.get("confidence", 0)
                print(f"   ✓ '{text[:30]}...'")
                print(f"     Predicted: {sentiment} (confidence: {confidence:.2%})")
                print(f"     Expected:  {expected}")
            else:
                print(f"   ✗ Failed: {data}")
                success = False

        except Exception as e:
            print(f"   ✗ Error: {e}")
            success = False

    return success


def test_batch_prediction(base_url: str) -> bool:
    """Test batch prediction endpoint."""
    print("\n3. Testing /predict/batch endpoint...")

    texts = [
        "ጥሩ ስራ ነው",
        "መጥፎ ነገር",
        "ፈጣሪ ይመስገን",
        "ሀገር አፍራሽ"
    ]

    try:
        response = requests.post(
            f"{base_url}/predict/batch",
            json={"texts": texts}
        )
        data = response.json()

        if response.status_code == 200:
            print(f"   Status: {response.status_code}")
            print(f"   Total predictions: {data.get('total_count')}")
            print(f"   Processing time: {data.get('processing_time_ms'):.2f}ms")
            print("   Results:")
            for result in data.get("results", []):
                print(f"     - {result['text'][:20]}... → {result['sentiment']} ({result['confidence']:.2%})")
            return True
        else:
            print(f"   ✗ Failed: {data}")
            return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test model info endpoint."""
    print("\n4. Testing /model/info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Model loaded: {data.get('loaded')}")
        print(f"   Model type: {data.get('model_type')}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Amharic Sentiment API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Amharic Sentiment Analysis API Test")
    print("=" * 50)
    print(f"Testing API at: {args.url}")

    results = {
        "health": test_health(args.url),
        "single_prediction": test_single_prediction(args.url),
        "batch_prediction": test_batch_prediction(args.url),
        "model_info": test_model_info(args.url)
    }

    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
