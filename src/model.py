"""
VGG-16 Transfer Learning Model — Cataract Binary Classification
Two-phase setup: (1) frozen base feature extraction, (2) top-block fine-tuning.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

from config import (
    INPUT_SHAPE, PHASE1_LR, PHASE2_LR
)


def build_vgg16_model(input_shape=INPUT_SHAPE, learning_rate=PHASE1_LR, freeze_base=True):
    """
    Build VGG-16 + custom head for binary cataract classification.

    freeze_base=True  → Phase 1: only custom head trains
    freeze_base=False → Phase 2: block4 + block5 also train
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    if freeze_base:
        base.trainable = False
    else:
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block4") or layer.name.startswith("block5")

    # ── Custom classification head ────────────────────────────────────────────
    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Block A
    x = layers.Dense(512, name="fc_a")(x)
    x = layers.BatchNormalization(name="bn_a")(x)
    x = layers.Activation("relu", name="relu_a")(x)
    x = layers.Dropout(0.5, name="drop_a")(x)

    # Block B
    x = layers.Dense(256, name="fc_b")(x)
    x = layers.BatchNormalization(name="bn_b")(x)
    x = layers.Activation("relu", name="relu_b")(x)
    x = layers.Dropout(0.4, name="drop_b")(x)

    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=base.input, outputs=output, name="VGG16_Cataract")
    _compile(model, learning_rate)
    return model


def unfreeze_top_blocks(model, learning_rate=PHASE2_LR):
    """Phase 2: unfreeze VGG-16 block4 and block5, lower the learning rate."""
    for layer in model.layers:
        if layer.name.startswith("block4") or layer.name.startswith("block5"):
            layer.trainable = True
    _compile(model, learning_rate)
    return model


def _compile(model, lr):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def print_trainable_summary(model):
    trainable   = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    frozen      = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    print(f"\n{'─'*55}")
    print(f"  Model            : {model.name}")
    print(f"  Total params     : {trainable + frozen:>12,}")
    print(f"  Trainable        : {trainable:>12,}")
    print(f"  Non-trainable    : {frozen:>12,}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    m = build_vgg16_model()
    print_trainable_summary(m)
    m.summary()
