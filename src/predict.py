"""
Single-image and batch inference for Cataract Detection.
Can be imported by the Flask app or run directly from the CLI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import tensorflow as tf

from config import (
    FINAL_MODEL_PATH, CHECKPOINT_PATH,
    IMAGE_SIZE, CLASS_NAMES, PREDICTION_THRESHOLD,
)


# ── Model loading (cached) ────────────────────────────────────────────────────

_model_cache = {}


def load_model(model_path=None):
    """Load (and cache) the trained model."""
    key = str(model_path or FINAL_MODEL_PATH)
    if key in _model_cache:
        return _model_cache[key]

    path = Path(key)
    if not path.exists():
        path = CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first.\n"
            f"Expected at: {FINAL_MODEL_PATH}"
        )

    print(f"  Loading model: {path}")
    model = tf.keras.models.load_model(str(path))
    _model_cache[key] = model
    return model


# ── Image preprocessing ───────────────────────────────────────────────────────

def to_array(image_input, target_size=IMAGE_SIZE):
    """
    Accept str/Path, PIL.Image, or numpy array.
    Returns (1, H, W, 3) float32 array normalised to [0, 1].
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Cannot read image: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
    else:
        raise TypeError(f"Unsupported type: {type(image_input)}")

    img = cv2.resize(img, target_size).astype(np.float32) / 255.0
    return np.expand_dims(img, 0)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_single(image_input, model=None, threshold=PREDICTION_THRESHOLD):
    """
    Returns:
        {class, label, probability, confidence, threshold}
    """
    if model is None:
        model = load_model()

    arr  = to_array(image_input)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = 1 if prob >= threshold else 0
    conf  = prob if label == 1 else 1.0 - prob

    return {
        "class":       CLASS_NAMES[label],
        "label":       label,
        "probability": round(prob, 4),
        "confidence":  round(conf * 100, 2),
        "threshold":   threshold,
    }


def predict_batch(image_paths, model=None, threshold=PREDICTION_THRESHOLD):
    """Predict for a list of image paths; errors are captured per image."""
    if model is None:
        model = load_model()

    results = []
    for path in image_paths:
        try:
            r = predict_single(path, model=model, threshold=threshold)
            r["image_path"] = str(path)
        except Exception as exc:
            r = {"image_path": str(path), "error": str(exc)}
        results.append(r)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Cataract Detection — CLI Prediction")
    parser.add_argument("image",      help="Path to a fundus image")
    parser.add_argument("--model",    default=None, help="Override model path")
    parser.add_argument("--threshold", type=float, default=PREDICTION_THRESHOLD)
    parser.add_argument("--json",     action="store_true", help="Print JSON output")
    args = parser.parse_args()

    model  = load_model(args.model)
    result = predict_single(args.image, model=model, threshold=args.threshold)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        bar = "█" * int(result["confidence"] / 5) + "░" * (20 - int(result["confidence"] / 5))
        print(f"\n{'═'*46}")
        print("  CATARACT DETECTION — PREDICTION RESULT")
        print(f"{'═'*46}")
        print(f"  Image      : {args.image}")
        print(f"  Prediction : {result['class'].upper()}")
        print(f"  Confidence : [{bar}] {result['confidence']}%")
        print(f"  Probability: {result['probability']}")
        print(f"{'═'*46}\n")
        if result["label"] == 1:
            print("  ⚠  Cataract detected. Please consult an ophthalmologist.\n")
        else:
            print("  ✓  No cataract detected. Eye appears normal.\n")


if __name__ == "__main__":
    _cli()
