import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet_mask(
    input_shape=(224, 224, 3),
    num_classes=3
):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained backbone
    base_model.trainable = False

    inputs = layers.Input(
        shape=input_shape
    )

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        inputs
    )

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.4
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return models.Model(
        inputs,
        outputs,
        name="MobileNetV2_MaskClassifier"
    )