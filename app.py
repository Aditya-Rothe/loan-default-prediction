import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Loan Default Risk Dashboard",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("loan_default_model.pkl")

# -------------------------
# HEADER
# -------------------------
st.title("🏦 Loan Default Risk Assessment")
st.caption("Machine Learning–based credit risk evaluation system")

st.divider()

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Prediction",
    "📈 Model Insights",
    "ℹ️ About Project"
])

# =========================
# TAB 1: PREDICTION
# =========================
with tab1:

    st.subheader("Applicant Information")

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

    st.divider()

    # -------------------------
    # RISK THRESHOLD SLIDER
    # -------------------------
    THRESHOLD = st.slider(
        "Risk Threshold (Bank Risk Appetite)",
        0.10, 0.60, 0.30, 0.05
    )

    # -------------------------
    # PREDICTION BUTTON
    # -------------------------
    if st.button("🔍 Assess Loan Risk"):

        # Education encoding
        education_map = {
            "High School": 0,
            "Bachelor's": 1,
            "Master's": 2,
            "PhD": 3
        }

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
            "Education": education_map[education],
        }

        # Initialize categorical features
        for feature in model.feature_names_in_:
            if "_" in feature:
                data[feature] = 0

        # Apply one-hot encoding logic
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

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Model prediction
        probability = model.predict_proba(input_df)[0][1]

        # Risk bands
        if probability < 0.20:
            risk_label = "🟢 Low Risk"
            color = "success"
        elif probability < THRESHOLD:
            risk_label = "🟡 Medium Risk"
            color = "warning"
        else:
            risk_label = "🔴 High Risk"
            color = "error"

        st.divider()

        # Result display
        getattr(st, color)(
            f"{risk_label}\n\n"
            f"Default Probability: {probability:.2%}\n"
            f"Decision Threshold: {THRESHOLD}"
        )

        # Probability bar
        st.progress(probability)

        # Applicant summary
        st.subheader("Applicant Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Credit Score", credit_score)
        col2.metric("DTI Ratio", dti_ratio)
        col3.metric("Loan Amount", loan_amount)

# =========================
# TAB 2: MODEL INSIGHTS
# =========================
with tab2:

    st.subheader("Top Feature Importances")

    importances = model.feature_importances_
    features = model.feature_names_in_

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(imp_df.set_index("Feature"))

    st.info("""
    The chart above shows the most influential features
    used by the Gradient Boosting model to assess loan default risk.
    """)

# =========================
# TAB 3: ABOUT PROJECT
# =========================
with tab3:

    st.subheader("Project Overview")

    st.markdown("""
    **Model:** Gradient Boosting Classifier  
    **Dataset:** Loan default dataset (Imbalanced)  
    **Default Rate:** ~12%  
    **Primary Metric:** ROC-AUC & Recall  
    **Threshold Tuning:** Business-driven risk sensitivity  

    **Objective:**  
    To predict the probability of loan default and assist
    financial institutions in making risk-aware lending decisions.
    """)

    st.success("Developed as an end-to-end Machine Learning deployment project 🚀")
