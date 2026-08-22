import os
import pandas as pd
import tensorflow as tf

from src.models.mobilenet_mask import (
    build_mobilenet_mask
)

from src.data.mobilenet_dataset import (
    load_mobilenet_datasets,
    preprocess_mobilenet_datasets
)

from src.data.mask_dataset import (
    get_class_weights
)


EPOCHS = 15

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -------------------------
# Dataset
# -------------------------

train_ds, val_ds, test_ds = (
    load_mobilenet_datasets()
)

train_ds, val_ds, test_ds = (
    preprocess_mobilenet_datasets(
        train_ds,
        val_ds,
        test_ds
    )
)


# -------------------------
# Class weights
# -------------------------

class_weights = get_class_weights()

print("Class weights:")
print(class_weights)


# -------------------------
# Model
# -------------------------

model = build_mobilenet_mask()

model.summary()


# -------------------------
# Compile
# -------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=(
        tf.keras.losses
        .SparseCategoricalCrossentropy()
    ),

    metrics=[
        "accuracy"
    ]
)


# -------------------------
# Callbacks
# -------------------------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        "models/mobilenet_mask_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# -------------------------
# Train
# -------------------------

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# -------------------------
# Save
# -------------------------

model.save(
    "models/mobilenet_mask_final.keras"
)

pd.DataFrame(
    history.history
).to_csv(
    "outputs/mobilenet_mask_history.csv",
    index=False
)

print("MobileNetV2 training complete.")