from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "mask",
    "no_mask",
    "improper_mask"
]

DATASET_DIR = Path(
    "data/processed/mask_classification"
)


def load_mobilenet_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "validation",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds, val_ds, test_ds


def preprocess_mobilenet_datasets(
    train_ds,
    val_ds,
    test_ds
):

    augmentation = tf.keras.Sequential([

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
            0.05,
            0.05
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )
    ])

    train_ds = train_ds.map(
        lambda x, y: (
            augmentation(x),
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

    return (
        train_ds,
        val_ds,
        test_ds
    )