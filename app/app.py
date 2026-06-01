"""
Flask Web Application — Cataract Disease Detection
Endpoints:
  GET  /          → index page
  POST /predict   → returns JSON prediction + Grad-CAM
  GET  /health    → liveness check
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import io
import base64
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config import IMAGE_SIZE, CLASS_NAMES
from src.predict import load_model, predict_single
from src.evaluate import compute_gradcam

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB limit
UPLOAD_DIR = Path(__file__).parent / "static" / "uploads"
ALLOWED = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# Lazy model singleton
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _ext_ok(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def _to_b64(arr: np.ndarray) -> str:
    """Convert a uint8 numpy image to a base64 PNG data URI."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _ext_ok(f.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED))}"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fname    = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
    filepath = UPLOAD_DIR / fname

    try:
        f.save(str(filepath))
        model  = get_model()

        # Prediction
        result = predict_single(filepath, model=model)

        # Original image (resized, base64)
        orig = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, IMAGE_SIZE)
        orig_b64 = _to_b64(orig)

        # Grad-CAM
        try:
            _, overlay = compute_gradcam(model, filepath)
            cam_b64 = _to_b64(overlay)
        except Exception:
            cam_b64 = None

        filepath.unlink(missing_ok=True)   # clean up temp file

        msg = (
            "⚠ Cataract detected. Please consult a qualified ophthalmologist."
            if result["label"] == 1
            else "✓ No cataract detected. Eye appears clinically normal."
        )

        return jsonify({
            "prediction":     result["class"],
            "label":          result["label"],
            "confidence":     result["confidence"],
            "probability":    result["probability"],
            "message":        msg,
            "original_image": orig_b64,
            "gradcam_image":  cam_b64,
        })

    except Exception as exc:
        filepath.unlink(missing_ok=True)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 55)
    print("  CataractAI — Flask Web Server")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)
