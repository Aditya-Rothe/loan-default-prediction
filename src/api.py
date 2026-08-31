import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = "models/loan_default_pipeline.pkl"
THRESHOLD = 0.20


app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk.",
    version="1.0.0",
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class LoanApplication(BaseModel):

    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Loan Default Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(application: LoanApplication):

    try:

        # Convert request to DataFrame
        input_data = pd.DataFrame(
            [application.model_dump()]
        )

        # Predict probability
        probability = model.predict_proba(
            input_data
        )[0][1]

        # Apply business threshold
        prediction = int(
            probability >= THRESHOLD
        )

        # Risk classification
        if probability >= 0.50:
            risk_level = "Very High Risk"
        elif probability >= THRESHOLD:
            risk_level = "High Risk"
        else:
            risk_level = "Low Risk"

        return {
            "default_probability": round(
                float(probability),
                4,
            ),
            "prediction": prediction,
            "risk_level": risk_level,
            "threshold": THRESHOLD,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )