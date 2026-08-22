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

from src.data.mask_dataset import (
    load_mask_datasets,
    preprocess_dataset,
    CLASS_NAMES
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/mask_cnn_best.keras"
)

OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

print("Loading best CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Load test dataset
# ============================================================

print("\nLoading test dataset...")

train_ds, val_ds, test_ds = (
    load_mask_datasets()
)

_, _, test_ds = preprocess_dataset(
    train_ds,
    val_ds,
    test_ds
)


# ============================================================
# Generate predictions
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []


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


# ============================================================
# Overall metrics
# ============================================================

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
print("CNN TEST RESULTS")
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


# ============================================================
# Classification report
# ============================================================

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


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)


# ============================================================
# Save results
# ============================================================

results_path = (
    OUTPUT_DIR /
    "mask_cnn_evaluation.txt"
)

with open(
    results_path,
    "w"
) as f:

    f.write(
        "CNN TEST RESULTS\n"
    )

    f.write(
        "================\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=====================\n\n"
    )

    f.write(report)

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        "================\n\n"
    )

    f.write(
        str(cm)
    )


print(
    f"\nResults saved to: {results_path}"
)