import tensorflow as tf
from tensorflow.keras import layers, models


def build_mask_cnn(
    input_shape=(224, 224, 3),
    num_classes=3
):

    model = models.Sequential([

        # -------------------------
        # Input
        # -------------------------

        layers.Input(shape=input_shape),

        # -------------------------
        # Block 1
        # -------------------------

        layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # -------------------------
        # Block 2
        # -------------------------

        layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # -------------------------
        # Block 3
        # -------------------------

        layers.Conv2D(
            128,
            (3, 3),
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # -------------------------
        # Block 4
        # -------------------------

        layers.Conv2D(
            256,
            (3, 3),
            padding="same"
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        # -------------------------
        # Feature aggregation
        # -------------------------

        layers.GlobalAveragePooling2D(),

        # -------------------------
        # Classification head
        # -------------------------

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model