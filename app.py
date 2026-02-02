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

if st.button("🔍 Predict Default Risk"):

    # Create RAW input dict (before encoding)
    input_dict = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    input_df = pd.DataFrame([input_dict])

    # 🔥 VERY IMPORTANT
    # Align columns with model
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Risk of Loan Default\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk – Loan Likely Safe\n\nProbability: {probability:.2%}")
