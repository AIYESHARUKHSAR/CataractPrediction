"""
Evaluation Pipeline — Confusion Matrix, ROC-AUC, Classification Report, Grad-CAM.
Run after training: python src/evaluate.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    RESULTS_DIR, PLOTS_DIR, IMAGE_SIZE, BATCH_SIZE,
    CLASS_NAMES, PREDICTION_THRESHOLD, GRADCAM_LAYER,
)
from src.predict import load_model, to_array, predict_single
from src.preprocess import parse_odir_annotations, split_dataset


# ── Test-set evaluation ───────────────────────────────────────────────────────

def evaluate_on_generator(model, gen):
    print("\nEvaluating on test set …")
    scores = model.evaluate(gen, verbose=1)
    metrics = {name: round(float(v), 4) for name, v in zip(model.metrics_names, scores)}
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    return metrics


def collect_predictions(model, gen):
    y_prob = model.predict(gen, verbose=1).flatten()
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
    y_true = gen.classes
    return y_true, y_pred, y_prob


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"Confusion Matrix  (Accuracy: {acc:.2%})", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return cm


def plot_roc_curve(y_true, y_prob, save_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, "#2196F3", lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "gray", lw=1.5, linestyle="--", label="Random")
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color="red", s=100, zorder=5,
               label=f"Optimal threshold: {opt_thr:.3f}")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Cataract Detection", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return roc_auc, opt_thr


def print_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nClassification Report\n" + "─" * 55)
    print(report)
    return report


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(model, image_input, layer_name=GRADCAM_LAYER):
    """
    Returns (heatmap HxW, superimposed HxWx3 uint8).
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    img_arr = to_array(image_input)      # (1, H, W, 3)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_arr)
        loss = preds[:, 0]

    grads        = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam          = conv_out[0] @ pooled_grads[..., tf.newaxis]
    cam          = tf.squeeze(cam).numpy()
    cam          = np.maximum(cam, 0)
    cam          = cam / (cam.max() + 1e-8)

    cam_resized  = cv2.resize(cam, IMAGE_SIZE)
    cam_colored  = (mpl_cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)

    if isinstance(image_input, (str, Path)):
        orig = cv2.cvtColor(cv2.imread(str(image_input)), cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, IMAGE_SIZE)
    else:
        orig = (img_arr[0] * 255).astype(np.uint8)

    superimposed = cv2.addWeighted(orig, 0.6, cam_colored, 0.4, 0)
    return cam, superimposed


def visualize_gradcam(model, image_paths, save_path=None):
    n = min(len(image_paths), 6)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, path in enumerate(image_paths[:n]):
        orig = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, IMAGE_SIZE)
        cam, overlay = compute_gradcam(model, path)
        res = predict_single(path, model=model)

        axes[i][0].imshow(orig);      axes[i][0].set_title("Original",       fontsize=10); axes[i][0].axis("off")
        axes[i][1].imshow(cam, cmap="jet"); axes[i][1].set_title("Grad-CAM Heatmap", fontsize=10); axes[i][1].axis("off")
        axes[i][2].imshow(overlay);   axes[i][2].set_title(f"{res['class']} ({res['confidence']}%)", fontsize=10); axes[i][2].axis("off")

    plt.suptitle("Grad-CAM Visualisation — Cataract Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_evaluation():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model = load_model()

    test_csv = RESULTS_DIR / "test_set.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
    else:
        df = parse_odir_annotations()
        _, _, test_df = split_dataset(df)

    gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
        dataframe=test_df, x_col="image_path", y_col="class_name",
        target_size=IMAGE_SIZE, class_mode="binary", batch_size=BATCH_SIZE,
        shuffle=False, classes=CLASS_NAMES,
    )

    metrics = evaluate_on_generator(model, gen)
    y_true, y_pred, y_prob = collect_predictions(model, gen)

    plot_confusion_matrix(y_true, y_pred, PLOTS_DIR / "confusion_matrix.png")
    roc_auc, opt_thr = plot_roc_curve(y_true, y_prob, PLOTS_DIR / "roc_curve.png")
    report = print_report(y_true, y_pred)

    (RESULTS_DIR / "classification_report.txt").write_text(report)

    samples = test_df[test_df["label"] == 1]["image_path"].values[:4].tolist()
    if samples:
        visualize_gradcam(model, samples, PLOTS_DIR / "gradcam_cataract.png")

    print(f"\n  ROC-AUC: {roc_auc:.4f}  |  Optimal threshold: {opt_thr:.3f}")
    print(f"  Results saved to: {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    run_evaluation()
