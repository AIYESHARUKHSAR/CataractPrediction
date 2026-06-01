"""
Two-Phase Training Pipeline
Phase 1 — frozen VGG-16 base, train custom head (10 epochs, lr=1e-4)
Phase 2 — unfreeze block4 + block5, fine-tune (20 epochs, lr=1e-5)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

from config import (
    MODELS_DIR, LOGS_DIR, RESULTS_DIR, PLOTS_DIR, TENSORBOARD_DIR,
    CHECKPOINT_PATH, FINAL_MODEL_PATH,
    PHASE1_EPOCHS, PHASE2_EPOCHS, PHASE1_LR, PHASE2_LR,
)
from src.model import build_vgg16_model, unfreeze_top_blocks, print_trainable_summary
from src.preprocess import (
    parse_odir_annotations, split_dataset,
    get_class_weights, create_data_generators,
)
from src.utils import plot_training_history, save_history


# ── Callback factory ──────────────────────────────────────────────────────────

def make_callbacks(phase: int) -> list:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = str(CHECKPOINT_PATH).replace(".h5", f"_ph{phase}.h5")

    return [
        ModelCheckpoint(
            filepath=ckpt, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc", patience=5, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        TensorBoard(
            log_dir=str(TENSORBOARD_DIR / f"ph{phase}_{ts}"),
            histogram_freq=1,
        ),
    ]


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_phase1(model, train_gen, val_gen, class_weights):
    print("\n" + "═" * 60)
    print("  PHASE 1 — Custom head training  (VGG-16 base FROZEN)")
    print(f"  Epochs: {PHASE1_EPOCHS}  |  LR: {PHASE1_LR}")
    print("═" * 60)
    print_trainable_summary(model)

    return model.fit(
        train_gen,
        epochs=PHASE1_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=make_callbacks(1),
        verbose=1,
    )


def run_phase2(model, train_gen, val_gen, class_weights):
    print("\n" + "═" * 60)
    print("  PHASE 2 — Fine-tuning block4 + block5  (LR lowered 10×)")
    print(f"  Epochs: {PHASE2_EPOCHS}  |  LR: {PHASE2_LR}")
    print("═" * 60)
    model = unfreeze_top_blocks(model)
    print_trainable_summary(model)

    return model, model.fit(
        train_gen,
        epochs=PHASE2_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=make_callbacks(2),
        verbose=1,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def train():
    print("═" * 60)
    print("  CNN-Based Cataract Detection — VGG-16 Training")
    print("  Dataset: ODIR-5K  |  Task: Binary Classification")
    print("═" * 60)

    for d in [MODELS_DIR, LOGS_DIR, RESULTS_DIR, PLOTS_DIR, TENSORBOARD_DIR]:
        os.makedirs(d, exist_ok=True)

    # Data
    df = parse_odir_annotations()
    train_df, val_df, test_df = split_dataset(df)
    test_df.to_csv(RESULTS_DIR / "test_set.csv", index=False)

    class_weights  = get_class_weights(train_df["label"].values)
    train_gen, val_gen, _ = create_data_generators(train_df, val_df, test_df)

    # Model
    model = build_vgg16_model(freeze_base=True, learning_rate=PHASE1_LR)

    # Phase 1
    h1 = run_phase1(model, train_gen, val_gen, class_weights)
    save_history(h1, RESULTS_DIR / "history_phase1.json")
    plot_training_history(h1, phase=1, save_dir=PLOTS_DIR)

    # Phase 2
    model, h2 = run_phase2(model, train_gen, val_gen, class_weights)
    save_history(h2, RESULTS_DIR / "history_phase2.json")
    plot_training_history(h2, phase=2, save_dir=PLOTS_DIR)

    # Persist
    model.save(str(FINAL_MODEL_PATH))
    print(f"\n  Final model saved → {FINAL_MODEL_PATH}")

    return model, h1, h2


if __name__ == "__main__":
    train()
