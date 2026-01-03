import streamlit as st
import numpy as np
import pickle

# Load model
with open("loan_default_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Loan Default Prediction App")

st.write("Enter applicant details")

age = st.number_input("Age", 18, 100)
income = st.number_input("Income", 0)
loan_amount = st.number_input("Loan Amount", 0)
credit_score = st.number_input("Credit Score", 300, 850)
months_employed = st.number_input("Months Employed", 0)
num_credit_lines = st.number_input("Number of Credit Lines", 0)
interest_rate = st.number_input("Interest Rate (%)", 0.0)
loan_term = st.number_input("Loan Term (months)", 0)
dti_ratio = st.number_input("Debt-to-Income Ratio", 0.0)

if st.button("Predict"):
    input_data = np.array([[age, income, loan_amount, credit_score,
                             months_employed, num_credit_lines,
                             interest_rate, loan_term, dti_ratio]])
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Loan Default Likely")
    else:
        st.success("✅ Low Risk: Loan Repayment Likely")
