import os

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/loan_default.csv"
MODEL_PATH = "models/loan_default_pipeline.pkl"

RANDOM_STATE = 42


def load_data():
    """Load the processed dataset."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def load_model():
    """Load the trained ML pipeline."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of default
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, y_pred))

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


def main():

    # Load dataset
    df = load_data()

    # Separate features and target
    X = df.drop(columns=["LoanID", "Default"])
    y = df["Default"]

    # Recreate the same test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Load model
    model = load_model()

    # Evaluate
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()