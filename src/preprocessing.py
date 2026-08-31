import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


def split_features_target(df):
    """
    Separate input features (X) and target variable (y).
    """

    X = df.drop(columns=["LoanID", "Default"])
    y = df["Default"]

    return X, y


def create_preprocessor():
    """
    Create the preprocessing pipeline.

    Numerical features:
        StandardScaler

    Categorical features:
        OneHotEncoder
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def prepare_data(df):
    """
    Prepare features, target, and preprocessing pipeline.
    """

    X, y = split_features_target(df)

    preprocessor = create_preprocessor()

    return X, y, preprocessor


if __name__ == "__main__":
    DATA_PATH = "data/processed/loan_default.csv"

    df = pd.read_csv(DATA_PATH)

    X, y, preprocessor = prepare_data(df)

    X_transformed = preprocessor.fit_transform(X)

    print("Original feature shape:", X.shape)
    print("Transformed feature shape:", X_transformed.shape)