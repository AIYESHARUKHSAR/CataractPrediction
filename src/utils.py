"""
Utility functions — training history plotting, directory setup, sample visualisation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    PLOTS_DIR, TENSORBOARD_DIR, IMAGE_SIZE, CLASS_NAMES,
)


# ── Training history ──────────────────────────────────────────────────────────

def plot_training_history(history, phase: int = 1, save_dir=None):
    """Plot loss / accuracy / AUC curves for a Keras History object."""
    keys   = [k for k in ("loss", "accuracy", "auc") if k in history.history]
    n_cols = len(keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, metric in zip(axes, keys):
        tr  = history.history[metric]
        val = history.history.get(f"val_{metric}", [])
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr,  "b-o", lw=2, ms=5, label="Train")
        if val:
            ax.plot(epochs, val, "r-o", lw=2, ms=5, label="Validation")
        ax.set_title(f"Phase {phase} — {metric.title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"Training History — Phase {phase}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = Path(save_dir) / f"history_phase{phase}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.show()


def save_history(history, filepath):
    os.makedirs(Path(filepath).parent, exist_ok=True)
    data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    Path(filepath).write_text(json.dumps(data, indent=2))
    print(f"  History saved: {filepath}")


def load_history(filepath):
    return json.loads(Path(filepath).read_text())


# ── Dataset visualisation ─────────────────────────────────────────────────────

def display_sample_images(df: pd.DataFrame, n: int = 8, save_path=None):
    """Show n random fundus images with their class labels."""
    normal   = df[df["label"] == 0].sample(min(n // 2, len(df[df["label"] == 0])), random_state=0)
    cataract = df[df["label"] == 1].sample(min(n // 2, len(df[df["label"] == 1])), random_state=0)
    samples  = pd.concat([normal, cataract]).sample(frac=1, random_state=0)

    cols = 4
    rows = max(1, (len(samples) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = axes.flatten() if rows > 1 else axes.reshape(-1)

    for ax, (_, row) in zip(axes_flat, samples.iterrows()):
        img = cv2.imread(str(row["image_path"]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            ax.imshow(img)
        color = "#e74c3c" if row["label"] == 1 else "#27ae60"
        ax.set_title(row["class_name"], color=color, fontsize=11, fontweight="bold")
        ax.axis("off")

    for ax in axes_flat[len(samples):]:
        ax.axis("off")

    plt.suptitle("ODIR-5K Sample Fundus Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(df: pd.DataFrame, save_path=None):
    counts = df["label"].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in counts.index]
    colors = ["#27ae60", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(cnt), ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Class Distribution — ODIR-5K Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Directory bootstrap ───────────────────────────────────────────────────────

def ensure_project_dirs():
    for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, PLOTS_DIR, TENSORBOARD_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Project directories are ready.")


if __name__ == "__main__":
    ensure_project_dirs()
