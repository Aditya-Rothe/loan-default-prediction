import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="centered"
)

# Load model
model = joblib.load("loan_default_model.pkl")

st.title("🏦 Loan Default Prediction System")
st.write("Predict whether a loan applicant is likely to default")

st.divider()

# -------------------------
# USER INPUTS
# -------------------------

age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income", 15000, 200000, 80000)
loan_amount = st.number_input("Loan Amount", 5000, 300000, 100000)
credit_score = st.number_input("Credit Score", 300, 900, 650)
months_employed = st.number_input("Months Employed", 0, 500, 60)
num_credit_lines = st.number_input("Number of Credit Lines", 0, 50, 5)
interest_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 12.5)
loan_term = st.number_input("Loan Term (months)", 6, 360, 60)
dti_ratio = st.number_input("DTI Ratio", 0.0, 1.0, 0.4)

education = st.selectbox(
    "Education",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

employment_type = st.selectbox(
    "Employment Type",
    ["Full-time", "Part-time", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Home", "Auto", "Education", "Business", "Other"]
)
has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

# -------------------------
# PREDICTION
# -------------------------

# Prediction
if st.button("🔍 Predict Default Risk"):

    # Ordinal Encoding (Education)
    education_map = {
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }

    education_encoded = education_map[education]

    # Base numerical features
    data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education_encoded,
    }

    # Initialize all one-hot encoded columns to 0
    categorical_columns = [
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner"
    ]

    for feature in model.feature_names_in_:
        if "_" in feature:
            data[feature] = 0

    # Set selected category = 1 (drop_first=True logic)
    if employment_type != "Full-time":
        data[f"EmploymentType_{employment_type}"] = 1

    if marital_status != "Single":
        data[f"MaritalStatus_{marital_status}"] = 1

    if has_mortgage == "Yes":
        data["HasMortgage_Yes"] = 1

    if has_dependents == "Yes":
        data["HasDependents_Yes"] = 1

    if loan_purpose != "Home":
        data[f"LoanPurpose_{loan_purpose}"] = 1

    if has_cosigner == "Yes":
        data["HasCoSigner_Yes"] = 1

    # Create DataFrame
    input_df = pd.DataFrame([data])

    # Align EXACT feature order
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # -------------------------
    # MODEL PREDICTION
    # -------------------------
    probability = model.predict_proba(input_df)[0][1]

    THRESHOLD = 0.30
    prediction = 1 if probability >= THRESHOLD else 0

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Loan Default\n\n"
            f"Default Probability: {probability:.2%}\n"
            f"Risk Threshold: {THRESHOLD}"
        )
    else:
        st.success(
            f"✅ Low Risk – Loan Likely Safe\n\n"
            f"Default Probability: {probability:.2%}\n"
            f"Risk Threshold: {THRESHOLD}"
        )

