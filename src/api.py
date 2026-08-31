
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = "models/loan_default_pipeline.pkl"
THRESHOLD = 0.20


# --------------------------------------------------
# Model Loading
# --------------------------------------------------

model = None


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    model = load_model()

    print("Model loaded successfully.")

    yield

    model = None

    print("Model unloaded.")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk.",
    version="1.0.0",
    lifespan=lifespan,
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

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:

        input_data = pd.DataFrame(
            [application.model_dump()]
        )

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= THRESHOLD
        )

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

