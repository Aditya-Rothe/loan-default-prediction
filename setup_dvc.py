import os
import subprocess
import streamlit as st

MODEL_PATH = "models/loan_default_pipeline.pkl"


def ensure_model():
    """
    Pull the model from DAGsHub using DVC if it is not
    already available locally.
    """

    # If model already exists, nothing to do.
    if os.path.exists(MODEL_PATH):
        return

    # Check Streamlit secrets
    if "DAGSHUB_USERNAME" not in st.secrets:
        raise RuntimeError("DAGSHUB_USERNAME is missing from Streamlit Secrets.")

    if "DAGSHUB_TOKEN" not in st.secrets:
        raise RuntimeError("DAGSHUB_TOKEN is missing from Streamlit Secrets.")

    username = st.secrets["DAGSHUB_USERNAME"]
    token = st.secrets["DAGSHUB_TOKEN"]

    try:
        # Configure DVC authentication locally on the Streamlit server
        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                "origin",
                "--local",
                "auth",
                "basic",
            ],
            check=True,
        )

        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                "origin",
                "--local",
                "user",
                username,
            ],
            check=True,
        )

        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                "origin",
                "--local",
                "password",
                token,
            ],
            check=True,
        )

        # Pull the model from DAGsHub
        subprocess.run(
            [
                "dvc",
                "pull",
                "-r",
                "origin",
                "models/loan_default_pipeline.pkl",
            ],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "DVC model pull failed. Please check DAGsHub credentials "
            "and DVC remote configuration."
        ) from e

    # Final verification
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "DVC pull completed, but the model file was not found."
        )