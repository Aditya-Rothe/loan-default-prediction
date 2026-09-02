import streamlit as st
import pandas as pd
import joblib
import os

# ==================================================

# PAGE CONFIGURATION

# ==================================================

st.set_page_config(
page_title="Loan Default Prediction",
page_icon="🏦",
layout="wide",
)

# ==================================================

# CONSTANTS

# ==================================================

MODEL_PATH = "models/loan_default_pipeline.pkl"
THRESHOLD = 0.20

# ==================================================

# LOAD MODEL

# ==================================================

@st.cache_resource
def load_model():

```
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found at: {MODEL_PATH}"
    )
    return None

return joblib.load(MODEL_PATH)
```

model = load_model()

# ==================================================

# HEADER

# ==================================================

st.title("🏦 Loan Default Risk Prediction")
st.markdown(
"""
Predict the probability of a customer defaulting on a loan using a
Machine Learning model built with an end-to-end MLOps pipeline.
"""
)

st.divider()

# ==================================================

# SIDEBAR

# ==================================================

st.sidebar.header("📋 Applicant Information")

st.sidebar.markdown(
"Enter the applicant's financial and personal information."
)

# ==================================================

# USER INPUTS

# ==================================================

col1, col2, col3 = st.columns(3)

with col1:

```
st.subheader("👤 Personal Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
)

education = st.selectbox(
    "Education",
    [
        "High School",
        "Bachelor's",
        "Master's",
        "PhD",
    ],
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced",
    ],
)

has_dependents = st.selectbox(
    "Has Dependents",
    [
        "Yes",
        "No",
    ],
)
```

with col2:

```
st.subheader("💼 Employment & Income")

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0,
)

employment_type = st.selectbox(
    "Employment Type",
    [
        "Full-time",
        "Part-time",
        "Self-employed",
        "Unemployed",
    ],
)

months_employed = st.number_input(
    "Months Employed",
    min_value=0,
    value=60,
)

num_credit_lines = st.number_input(
    "Number of Credit Lines",
    min_value=0,
    value=3,
)
```

with col3:

```
st.subheader("💰 Loan & Financial Information")

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=150000.0,
)

credit_score = st.number_input(
    "Credit Score",
    min_value=0.0,
    value=650.0,
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=10.5,
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=36,
)

dti_ratio = st.number_input(
    "Debt-to-Income Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
)
```

st.divider()

# ==================================================

# ADDITIONAL INFORMATION

# ==================================================

st.subheader("🏠 Additional Information")

col4, col5, col6 = st.columns(3)

with col4:

```
has_mortgage = st.selectbox(
    "Has Mortgage",
    [
        "Yes",
        "No",
    ],
)
```

with col5:

```
loan_purpose = st.selectbox(
    "Loan Purpose",
    [
        "Auto",
        "Business",
        "Education",
        "Home",
        "Other",
    ],
)
```

with col6:

```
has_cosigner = st.selectbox(
    "Has Co-Signer",
    [
        "Yes",
        "No",
    ],
)
```

st.divider()

# ==================================================

# PREDICTION

# ==================================================

if st.button(
"🔍 Predict Loan Default Risk",
use_container_width=True,
):

```
if model is None:

    st.error(
        "Model could not be loaded."
    )

else:

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Income": [income],
            "LoanAmount": [loan_amount],
            "CreditScore": [credit_score],
            "MonthsEmployed": [months_employed],
            "NumCreditLines": [num_credit_lines],
            "InterestRate": [interest_rate],
            "LoanTerm": [loan_term],
            "DTIRatio": [dti_ratio],
            "Education": [education],
            "EmploymentType": [employment_type],
            "MaritalStatus": [marital_status],
            "HasMortgage": [has_mortgage],
            "HasDependents": [has_dependents],
            "LoanPurpose": [loan_purpose],
            "HasCoSigner": [has_cosigner],
        }
    )


    # Predict probability

    probability = model.predict_proba(
        input_data
    )[0][1]


    # Apply business threshold

    prediction = int(
        probability >= THRESHOLD
    )


    # ==================================================
    # RESULTS
    # ==================================================

    st.subheader("📊 Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Default Probability",
            f"{probability:.2%}",
        )


    with result_col2:

        st.metric(
            "Business Threshold",
            f"{THRESHOLD:.0%}",
        )


    with result_col3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Potential Default",
            )

        else:

            st.metric(
                "Prediction",
                "Low Risk",
            )


    st.divider()


    # ==================================================
    # RISK LEVEL
    # ==================================================

    if probability >= 0.50:

        st.error(
            "🔴 Very High Risk of Loan Default"
        )

    elif probability >= THRESHOLD:

        st.warning(
            "🟠 High Risk of Loan Default"
        )

    else:

        st.success(
            "🟢 Low Risk of Loan Default"
        )


    # ==================================================
    # BUSINESS EXPLANATION
    # ==================================================

    st.subheader("💡 Business Interpretation")

    if prediction == 1:

        st.write(
            f"""
            The predicted probability of loan default is
            **{probability:.2%}**.

            Since this probability is above the selected business
            threshold of **{THRESHOLD:.0%}**, the applicant is
            classified as a potential default risk.

            The application may require additional review before
            loan approval.
            """
        )

    else:

        st.write(
            f"""
            The predicted probability of loan default is
            **{probability:.2%}**.

            Since this probability is below the selected business
            threshold of **{THRESHOLD:.0%}**, the applicant is
            classified as relatively low risk.
            """
        )
```

# ==================================================

# FOOTER

# ==================================================

st.divider()

st.caption(
"""
Built by Aditya Rothe | End-to-End MLOps Loan Default Prediction Project
"""
)
