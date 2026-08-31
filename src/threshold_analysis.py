import os

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/loan_default.csv"
MODEL_PATH = "models/loan_default_pipeline.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20

THRESHOLDS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]


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


def analyze_thresholds(model, X_test, y_test):
    """Evaluate model performance at different thresholds."""

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n========== THRESHOLD ANALYSIS ==========\n")

    results = []

    for threshold in THRESHOLDS:

        # Convert probabilities into class predictions
        y_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return results_df


def main():

    # Load data
    df = load_data()

    # Separate features and target
    X = df.drop(columns=["LoanID", "Default"])
    y = df["Default"]

    # Recreate the same test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Load trained pipeline
    model = load_model()

    # Analyze thresholds
    analyze_thresholds(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()