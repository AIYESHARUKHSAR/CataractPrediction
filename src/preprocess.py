"""
ODIR-5K Dataset Preprocessing Pipeline
Parses the Excel annotation file, extracts Cataract / Normal labels,
applies augmentation for training, and returns Keras generators.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

from config import (
    ODIR_ANNOTATIONS, ODIR_IMAGES_DIR, CLASS_NAMES,
    IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    AUG_ROTATION_RANGE, AUG_ZOOM_RANGE, AUG_BRIGHTNESS_RANGE,
    AUG_HORIZONTAL_FLIP, AUG_FILL_MODE,
)


# ── Annotation Parsing ────────────────────────────────────────────────────────

def parse_odir_annotations(annotation_file=ODIR_ANNOTATIONS):
    """
    Parse ODIR-5K Excel file → DataFrame with columns:
      image_path | label | class_name | patient_id | eye | keywords
    Only rows that map cleanly to Normal (0) or Cataract (1) are kept.
    """
    print(f"Loading annotations: {annotation_file}")

    ext = str(annotation_file).lower()
    df = pd.read_excel(annotation_file) if ext.endswith((".xlsx", ".xls")) else pd.read_csv(annotation_file)

    print(f"  Columns : {df.columns.tolist()}")
    print(f"  Rows    : {len(df)}")

    records = []

    for _, row in df.iterrows():
        patient_id = str(row.get("ID", row.get("id", "")))

        for eye, img_col, diag_col in [
            ("Left",  "Left-Fundus",  "Left-Diagnostic Keywords"),
            ("Right", "Right-Fundus", "Right-Diagnostic Keywords"),
        ]:
            img_name = str(row.get(img_col, "")).strip()
            keywords = str(row.get(diag_col, "")).strip().lower()

            if not img_name or img_name == "nan":
                continue

            img_path = ODIR_IMAGES_DIR / img_name
            if not img_path.exists():
                continue

            if "cataract" in keywords:
                label = 1
            elif "normal fundus" in keywords or keywords in ("n", "normal"):
                label = 0
            else:
                continue  # skip ambiguous / multi-disease entries

            records.append({
                "image_path": str(img_path),
                "label":      label,
                "class_name": CLASS_NAMES[label],
                "patient_id": patient_id,
                "eye":        eye,
                "keywords":   keywords,
            })

    result = pd.DataFrame(records)

    print(f"\n  Usable samples : {len(result)}")
    print(f"    Normal       : {(result['label'] == 0).sum()}")
    print(f"    Cataract     : {(result['label'] == 1).sum()}")

    return result


# ── Dataset Split ─────────────────────────────────────────────────────────────

def split_dataset(df):
    """Stratified 70 / 15 / 15 train / val / test split."""
    train_val, test = train_test_split(
        df, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=RANDOM_SEED, stratify=train_val["label"]
    )

    print(f"\n  Train : {len(train)}  |  Val : {len(val)}  |  Test : {len(test)}")
    return train, val, test


# ── Class Weights ─────────────────────────────────────────────────────────────

def get_class_weights(labels):
    """Return {0: w0, 1: w1} to balance minority class during training."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    cw = dict(zip(classes.astype(int), weights))
    print(f"  Class weights: {cw}")
    return cw


# ── Keras Generators ──────────────────────────────────────────────────────────

def create_data_generators(train_df, val_df, test_df):
    """Return (train_gen, val_gen, test_gen) Keras ImageDataGenerators."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=AUG_ROTATION_RANGE,
        zoom_range=AUG_ZOOM_RANGE,
        brightness_range=AUG_BRIGHTNESS_RANGE,
        horizontal_flip=AUG_HORIZONTAL_FLIP,
        fill_mode=AUG_FILL_MODE,
        shear_range=0.08,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )
    plain_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    def _gen(datagen, df, shuffle):
        return datagen.flow_from_dataframe(
            dataframe=df,
            x_col="image_path",
            y_col="class_name",
            target_size=IMAGE_SIZE,
            color_mode="rgb",
            class_mode="binary",
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            seed=RANDOM_SEED,
            classes=CLASS_NAMES,
        )

    return (
        _gen(train_datagen, train_df, shuffle=True),
        _gen(plain_datagen, val_df,   shuffle=False),
        _gen(plain_datagen, test_df,  shuffle=False),
    )


# ── Single-image helper ───────────────────────────────────────────────────────

def preprocess_single(image_path, target_size=IMAGE_SIZE):
    """Load → RGB → resize → normalize → (1, H, W, 3) float32."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


if __name__ == "__main__":
    df = parse_odir_annotations()
    train, val, test = split_dataset(df)
    get_class_weights(train["label"].values)
