import os
import pandas as pd


DATA_PATH = "data/processed/loan_default.csv"

EXPECTED_COLUMNS = [
    "LoanID",
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
    "Default",
]


def load_data():
    """Load the processed dataset."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def validate_columns(df):
    """Validate dataset columns."""

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        missing_columns = set(EXPECTED_COLUMNS) - set(actual_columns)
        extra_columns = set(actual_columns) - set(EXPECTED_COLUMNS)

        raise ValueError(
            f"Column validation failed.\n"
            f"Missing columns: {missing_columns}\n"
            f"Extra columns: {extra_columns}"
        )

    print("✓ Column validation passed.")


def validate_missing_values(df):
    """Check for missing values."""

    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    print("✓ Missing value validation passed.")


def validate_duplicates(df):
    """Check for duplicate rows."""

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    print("✓ Duplicate validation passed.")


def validate_target(df):
    """Validate target variable."""

    target_values = set(df["Default"].unique())

    expected_values = {0, 1}

    if not target_values.issubset(expected_values):
        raise ValueError(
            f"Invalid target values found: {target_values}"
        )

    print("✓ Target validation passed.")


def validate_data(df):
    """Run all validation checks."""

    validate_columns(df)
    validate_missing_values(df)
    validate_duplicates(df)
    validate_target(df)

    print("\n✓ All data validation checks passed.")


def main():
    df = load_data()

    print(f"Dataset shape: {df.shape}\n")

    validate_data(df)


if __name__ == "__main__":
    main()