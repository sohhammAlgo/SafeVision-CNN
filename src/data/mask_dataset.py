from pathlib import Path

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "mask",
    "no_mask",
    "improper_mask"
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# Dataset paths
# ============================================================

DATASET_DIR = Path(
    "data/processed/mask_classification"
)

# ============================================================
# Data augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.08
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    tf.keras.layers.RandomContrast(
        0.10
    )
])


# ============================================================
# Load datasets
# ============================================================

def load_mask_datasets():
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"
    test_dir = DATASET_DIR / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(

        train_dir,

        labels="inferred",

        label_mode="int",

        class_names=CLASS_NAMES,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=True,

        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(

        val_dir,

        labels="inferred",

        label_mode="int",

        class_names=CLASS_NAMES,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(

        test_dir,

        labels="inferred",

        label_mode="int",

        class_names=CLASS_NAMES,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    return train_ds, val_ds, test_ds


# ============================================================
# Preprocessing
# ============================================================

def preprocess_dataset(
        train_ds,
        val_ds,
        test_ds
):
    normalization = tf.keras.layers.Rescaling(
        1.0 / 255
    )

    # Training:
    # augmentation + normalization

    train_ds = train_ds.map(
        lambda x, y: (
            normalization(
                data_augmentation(x)
            ),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Validation:
    # normalization only

    val_ds = val_ds.map(
        lambda x, y: (
            normalization(x),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Test:
    # normalization only

    test_ds = test_ds.map(
        lambda x, y: (
            normalization(x),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, val_ds, test_ds


# ============================================================
# Class weights
# ============================================================

def get_class_weights():
    class_counts = np.array([
        2237,  # mask
        462,  # no_mask
        84  # improper_mask
    ])

    total = class_counts.sum()

    class_weights = {}

    for class_id, count in enumerate(class_counts):
        class_weights[class_id] = (
                total /
                (NUM_CLASSES * count)
        )

    return class_weights