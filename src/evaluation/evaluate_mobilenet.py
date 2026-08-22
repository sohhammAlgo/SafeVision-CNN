from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data.mobilenet_dataset import (
    load_mobilenet_datasets
)


MODEL_PATH = Path(
    "models/mobilenet_mask_best.keras"
)

CLASS_NAMES = [
    "mask",
    "no_mask",
    "improper_mask"
]


print("Loading MobileNetV2...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


print("\nLoading test dataset...")

_, _, test_ds = load_mobilenet_datasets()


y_true = []
y_pred = []


print("\nGenerating predictions...")

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Overall metrics

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n================================")
print("MOBILENETV2 TEST RESULTS")
print("================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# Classification report

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# Confusion matrix

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)