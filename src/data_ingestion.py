import os
import pandas as pd


RAW_DATA_PATH = "data/raw/Loan_default.csv"
PROCESSED_DATA_PATH = "data/processed/loan_default.csv"


def load_data():
    """
    Load the raw loan default dataset.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    return df


def save_processed_data(df):
    """
    Save the dataset to the processed data directory.
    """
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")


def main():
    df = load_data()

    # For now, we are only ingesting the data.
    # Cleaning and preprocessing will happen in later stages.

    save_processed_data(df)


if __name__ == "__main__":
    main()