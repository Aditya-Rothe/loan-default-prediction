import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor, split_features_target


DATA_PATH = "data/processed/loan_default.csv"
MODEL_PATH = "models/loan_default_pipeline.pkl"
PARAMS_PATH = "params.yaml"


def load_data():
    """Load the processed dataset."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def load_params():
    """Load project parameters."""

    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(
            f"Parameters file not found at: {PARAMS_PATH}"
        )

    with open(PARAMS_PATH, "r") as file:
        return yaml.safe_load(file)


def create_model(random_state):
    """Create the Gradient Boosting classifier."""

    return GradientBoostingClassifier(
        random_state=random_state
    )


def train_model(X_train, y_train, random_state):
    """Create and train preprocessing + model pipeline."""

    preprocessor = create_preprocessor()
    model = create_model(random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate the trained model."""

    probabilities = model.predict_proba(X_test)[:, 1]

    y_pred = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


def main():

    # Load parameters
    params = load_params()

    random_state = params["model"]["random_state"]
    test_size = params["training"]["test_size"]
    threshold = params["training"]["threshold"]

    # Load data
    df = load_data()

    # Features and target
    X, y = split_features_target(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # MLflow experiment
    mlflow.set_experiment(
        "Loan Default Prediction"
    )

    with mlflow.start_run():

        # Train
        pipeline = train_model(
            X_train,
            y_train,
            random_state,
        )

        # Evaluate
        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
            threshold,
        )

        # Log parameters
        mlflow.log_params(
            {
                "model": params["model"]["name"],
                "random_state": random_state,
                "test_size": test_size,
                "threshold": threshold,
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save model locally
        os.makedirs(
            "models",
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            MODEL_PATH,
        )

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            name="loan_default_pipeline",
        )

        # Output
        print("\n========== MODEL EVALUATION ==========")

        for metric, value in metrics.items():
            print(
                f"{metric.capitalize():<10}: {value:.4f}"
            )

        print(
            f"\nModel saved to: {MODEL_PATH}"
        )

        print(
    "MLflow experiment logged successfully."
        )


if __name__ == "__main__":
    main()