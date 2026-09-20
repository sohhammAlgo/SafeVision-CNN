import cv2
import numpy as np
import tensorflow as tf


CLASS_NAMES = [
    "mask",
    "no_mask",
    "improper_mask"
]


class GradCAM:

    def __init__(
        self,
        model_path="models/mask_cnn_final.keras",
        target_layer="conv2d_3"
    ):

        self.model = tf.keras.models.load_model(
            model_path
        )

        self.target_layer = self.model.get_layer(
            target_layer
        )

    def generate(self, image):

        # ========================================
        # Prepare image
        # ========================================

        resized = cv2.resize(
            image,
            (224, 224)
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        input_image = (
            rgb.astype(np.float32) / 255.0
        )

        input_tensor = tf.convert_to_tensor(
            input_image[None, ...]
        )

        # ========================================
        # Gradient calculation
        # ========================================

        with tf.GradientTape() as tape:

            tape.watch(input_tensor)

            x = input_tensor

            # Run through model layer-by-layer
            for layer in self.model.layers:

                x = layer(x)

                if layer.name == self.target_layer.name:
                    conv_output = x

            predictions = x

            predicted_index = tf.argmax(
                predictions[0]
            )

            class_score = predictions[
                0,
                predicted_index
            ]

        gradients = tape.gradient(
            class_score,
            conv_output
        )

        if gradients is None:

            raise RuntimeError(
                "Gradients could not be calculated "
                "for the target layer."
            )

        # ========================================
        # Grad-CAM weights
        # ========================================

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_output = conv_output[0]

        heatmap = tf.reduce_sum(
            conv_output * pooled_gradients,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        heatmap = heatmap / (
            max_value + 1e-8
        )

        heatmap = heatmap.numpy()

        # ========================================
        # Resize heatmap
        # ========================================

        heatmap = cv2.resize(
            heatmap,
            (
                image.shape[1],
                image.shape[0]
            )
        )

        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # ========================================
        # Overlay
        # ========================================

        overlay = cv2.addWeighted(
            image,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ========================================
        # Prediction
        # ========================================

        probabilities = (
            predictions[0].numpy()
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        return (
            overlay,
            predicted_class,
            confidence,
            heatmap
        )