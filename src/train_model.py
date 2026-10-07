import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from data_preprocessing import (
    load_data,
    preprocess_data
)


# ==========================================
# Create directories
# ==========================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)


# ==========================================
# Load Dataset
# ==========================================

data_path = "../data/heart.csv"

df = load_data(data_path)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# Preprocess Data
# ==========================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = preprocess_data(df)


print("\nPreprocessing completed successfully.")

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# ==========================================
# Logistic Regression
# ==========================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# Decision Tree
# ==========================================

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

tree_predictions = decision_tree.predict(
    X_test
)

tree_probabilities = decision_tree.predict_proba(
    X_test
)[:, 1]


# ==========================================
# Evaluation Function
# ==========================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print("\n================================")
    print(model_name)
    print("================================")

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    return accuracy, precision, recall, f1


# ==========================================
# Evaluate Models
# ==========================================

logistic_scores = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions
)

tree_scores = evaluate_model(
    "Decision Tree",
    y_test,
    tree_predictions
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    logistic_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Disease",
        "Disease"
    ],
    yticklabels=[
        "No Disease",
        "Disease"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    "../outputs/confusion_matrix.png"
)

plt.close()


# ==========================================
# ROC Curve
# ==========================================

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

tree_auc = roc_auc_score(
    y_test,
    tree_probabilities
)

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_probabilities
)

fpr_tree, tpr_tree, _ = roc_curve(
    y_test,
    tree_probabilities
)


plt.figure(figsize=(7, 5))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression AUC = {logistic_auc:.2f}"
)

plt.plot(
    fpr_tree,
    tpr_tree,
    label=f"Decision Tree AUC = {tree_auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "../outputs/roc_curve.png"
)

plt.close()


# ==========================================
# Model Comparison
# ==========================================

models = [
    "Logistic Regression",
    "Decision Tree"
]

accuracies = [
    logistic_scores[0],
    tree_scores[0]
]

plt.figure(figsize=(7, 5))

plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy")

plt.xlabel("Model")

plt.title(
    "Model Accuracy Comparison"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "../outputs/model_comparison.png"
)

plt.close()


# ==========================================
# Save Models
# ==========================================

joblib.dump(
    logistic_model,
    "../models/logistic_model.pkl"
)

joblib.dump(
    decision_tree,
    "../models/decision_tree_model.pkl"
)

joblib.dump(
    preprocessor,
    "../models/preprocessor.pkl"
)


print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")

print(
    "\nLogistic Regression AUC:",
    round(logistic_auc, 4)
)

print(
    "Decision Tree AUC:",
    round(tree_auc, 4)
)

print("\nSaved files:")

print(
    "../models/logistic_model.pkl"
)

print(
    "../models/decision_tree_model.pkl"
)

print(
    "../models/preprocessor.pkl"
)

print("\nGenerated graphs:")

print(
    "../outputs/confusion_matrix.png"
)

print(
    "../outputs/roc_curve.png"
)

print(
    "../outputs/model_comparison.png"
)