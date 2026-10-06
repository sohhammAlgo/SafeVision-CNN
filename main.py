import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------------------------
# 1. Create Dummy Dataset
# -------------------------------------------------

np.random.seed(42)

n = 100

age = np.random.randint(21, 70, n)
bmi = np.round(np.random.uniform(18, 40, n), 1)
glucose = np.random.randint(70, 200, n)

# Generate dummy risk score
risk_score = (
    0.05 * (glucose - 100)
    + 0.8 * (bmi - 25)
    + 0.15 * (age - 40)
)

# 0 = Non-Diabetic
# 1 = Diabetic
outcome = (risk_score > 8).astype(int)

df = pd.DataFrame({
    "Age": age,
    "BMI": bmi,
    "Glucose": glucose,
    "Outcome": outcome
})


print("========== DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["Outcome"].value_counts())


# -------------------------------------------------
# 2. Separate Input and Output
# -------------------------------------------------

X = df[["Age", "BMI", "Glucose"]].values
y = df["Outcome"].values.reshape(-1, 1)


# -------------------------------------------------
# 3. Split Dataset
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# -------------------------------------------------
# 4. Normalize Input Data
# -------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------------------------------
# 5. Sigmoid Activation Function
# -------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# -------------------------------------------------
# 6. Create Neural Network
#    3 Input -> 3 Hidden -> 1 Output
# -------------------------------------------------

input_neurons = 3
hidden_neurons = 3
output_neurons = 1

np.random.seed(42)

# Input to Hidden Layer
W1 = np.random.randn(
    input_neurons,
    hidden_neurons
) * 0.1

b1 = np.zeros((1, hidden_neurons))

# Hidden to Output Layer
W2 = np.random.randn(
    hidden_neurons,
    output_neurons
) * 0.1

b2 = np.zeros((1, output_neurons))


# -------------------------------------------------
# 7. Training using Backpropagation
# -------------------------------------------------

learning_rate = 0.25
epochs = 10

print("\n========== TRAINING ==========")

for epoch in range(epochs):

    # ---------- Forward Propagation ----------

    hidden_input = np.dot(X_train, W1) + b1

    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(
        hidden_output,
        W2
    ) + b2

    output = sigmoid(output_input)


    # ---------- Calculate Error ----------

    error = y_train - output

    loss = np.mean(error ** 2)


    # ---------- Backpropagation ----------

    output_delta = (
        error *
        sigmoid_derivative(output)
    )

    hidden_error = np.dot(
        output_delta,
        W2.T
    )

    hidden_delta = (
        hidden_error *
        sigmoid_derivative(hidden_output)
    )


    # ---------- Update Hidden -> Output ----------

    W2 += learning_rate * np.dot(
        hidden_output.T,
        output_delta
    )

    b2 += learning_rate * np.sum(
        output_delta,
        axis=0,
        keepdims=True
    )


    # ---------- Update Input -> Hidden ----------

    W1 += learning_rate * np.dot(
        X_train.T,
        hidden_delta
    )

    b1 += learning_rate * np.sum(
        hidden_delta,
        axis=0,
        keepdims=True
    )


    # ---------- Display Epoch Results ----------

    print("\nEpoch:", epoch + 1)
    print("Loss:", round(loss, 4))

    print("\nInput-Hidden Weights:")
    print(np.round(W1, 4))

    print("\nHidden-Output Weights:")
    print(np.round(W2, 4))

    print("\nHidden Bias:")
    print(np.round(b1, 4))

    print("\nOutput Bias:")
    print(np.round(b2, 4))


# -------------------------------------------------
# 8. Testing the Trained Network
# -------------------------------------------------

hidden_test = sigmoid(
    np.dot(X_test, W1) + b1
)

output_test = sigmoid(
    np.dot(hidden_test, W2) + b2
)


# Classification threshold = 0.5

predictions = (
    output_test >= 0.5
).astype(int)


# -------------------------------------------------
# 9. Calculate Accuracy
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL RESULT ==========")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# -------------------------------------------------
# 10. Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# -------------------------------------------------
# 11. Classification Report
# -------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Non-Diabetic",
            "Diabetic"
        ]
    )
)


# -------------------------------------------------
# 12. New Patient Prediction
# -------------------------------------------------

print("\n========== NEW PATIENT PREDICTION ==========")

new_age = float(
    input("Enter Age: ")
)

new_bmi = float(
    input("Enter BMI: ")
)

new_glucose = float(
    input("Enter Glucose Level: ")
)


new_patient = np.array([
    [new_age, new_bmi, new_glucose]
])


# Scale new patient data

new_patient_scaled = scaler.transform(
    new_patient
)


# Forward propagation

hidden = sigmoid(
    np.dot(
        new_patient_scaled,
        W1
    ) + b1
)

probability = sigmoid(
    np.dot(
        hidden,
        W2
    ) + b2
)


# Classification

prediction = int(
    probability[0][0] >= 0.5
)


print(
    "\nDiabetes Probability:",
    round(
        float(probability[0][0]),
        4
    )
)


if prediction == 1:
    print("Prediction: DIABETIC")
else:
    print("Prediction: NON-DIABETIC")