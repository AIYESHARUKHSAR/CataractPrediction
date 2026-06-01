"""
Global configuration — all hyperparameters, paths, and constants.
Edit this file before training; do not scatter magic numbers in source files.
"""
import os
from pathlib import Path

# ── Base Directories ──────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent

DATA_DIR        = BASE_DIR / "data"
MODELS_DIR      = BASE_DIR / "models"
LOGS_DIR        = BASE_DIR / "logs"
RESULTS_DIR     = BASE_DIR / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

# ── ODIR-5K Dataset ──────────────────────────────────────────────────────────
ODIR_IMAGES_DIR  = DATA_DIR / "ODIR-5K_Training_Images"
ODIR_ANNOTATIONS = DATA_DIR / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
PROCESSED_DIR    = DATA_DIR / "processed"

# ── Model Checkpoints ─────────────────────────────────────────────────────────
CHECKPOINT_PATH  = MODELS_DIR / "vgg16_cataract_best.h5"
FINAL_MODEL_PATH = MODELS_DIR / "vgg16_cataract_final.h5"

# ── Image Configuration ───────────────────────────────────────────────────────
IMAGE_SIZE     = (224, 224)       # VGG-16 standard input
IMAGE_CHANNELS = 3
INPUT_SHAPE    = (224, 224, 3)

# ── Training Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 32
PHASE1_EPOCHS = 10    # Feature extraction — frozen VGG base
PHASE2_EPOCHS = 20    # Fine-tuning — block4 + block5 unfrozen
PHASE1_LR     = 1e-4
PHASE2_LR     = 1e-5

# ── Dataset Split ─────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# ── Classes (binary) ─────────────────────────────────────────────────────────
CLASS_NAMES = ["Normal", "Cataract"]  # index 0 → Normal, 1 → Cataract
NUM_CLASSES = 1                        # sigmoid output

# ── Data Augmentation ─────────────────────────────────────────────────────────
AUG_ROTATION_RANGE   = 15
AUG_ZOOM_RANGE       = 0.10
AUG_BRIGHTNESS_RANGE = (0.8, 1.2)
AUG_HORIZONTAL_FLIP  = True
AUG_FILL_MODE        = "nearest"

# ── Inference ─────────────────────────────────────────────────────────────────
PREDICTION_THRESHOLD = 0.5

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
GRADCAM_LAYER = "block5_conv3"   # Last conv layer of VGG-16
