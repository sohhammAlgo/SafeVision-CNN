import os

import tensorflow as tf

from src.models.mask_cnn import build_mask_cnn

from src.data.mask_dataset import (
    load_mask_datasets,
    preprocess_dataset,
    get_class_weights
)


# ============================================================
# Configuration
# ============================================================

EPOCHS = 30

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load dataset
# ============================================================

print("Loading datasets...")

train_ds, val_ds, test_ds = (
    load_mask_datasets()
)

train_ds, val_ds, test_ds = (
    preprocess_dataset(
        train_ds,
        val_ds,
        test_ds
    )
)


# ============================================================
# Class weights
# ============================================================

class_weights = get_class_weights()

print("\nClass weights:")
print(class_weights)


# ============================================================
# Build model
# ============================================================

print("\nBuilding CNN...")

model = build_mask_cnn()

model.summary()


# ============================================================
# Compile
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=(
            f"{MODEL_DIR}/"
            "mask_cnn_best.keras"
        ),

        monitor="val_loss",

        save_best_only=True,

        mode="min",

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=7,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1
    )
]


# ============================================================
# Train
# ============================================================

print("\nStarting training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# Save final model
# ============================================================

model.save(
    f"{MODEL_DIR}/mask_cnn_final.keras"
)


print("\nTraining complete.")

print(
    "Best model:",
    f"{MODEL_DIR}/mask_cnn_best.keras"
)

print(
    "Final model:",
    f"{MODEL_DIR}/mask_cnn_final.keras"
)


# ============================================================
# Save training history
# ============================================================

history_path = (
    f"{OUTPUT_DIR}/"
    "mask_cnn_history.csv"
)

import pandas as pd

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    history_path,
    index=False
)

print(
    "History saved:",
    history_path
)