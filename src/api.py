import os
import joblib
import pandas as pd
import streamlit as st

from setup_dvc import ensure_model


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

MODEL_PATH = "models/loan_default_pipeline.pkl"
THRESHOLD = 0.20


# --------------------------------------------------
# Ensure Model is Available
# --------------------------------------------------

ensure_model()


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("💰 Loan Default Prediction System")

st.markdown(
    """
    ### Predict Loan Default Risk

    This application predicts the probability that a loan applicant
    may default on their loan.

    The model uses a **0.20 probability threshold** to identify
    potentially risky applicants.
    """
)


# --------------------------------------------------
# Sidebar - Model Information
# --------------------------------------------------

with st.sidebar:
    st.header("📊 Model Information")

    st.write("**Model:** Gradient Boosting Classifier")
    st.write("**ROC-AUC:** 0.7580")
    st.write("**Decision Threshold:** 0.20")

    st.divider()

    st.info(
        """
        The default probability threshold was reduced from 0.50
        to 0.20 to improve recall and identify more potential
        loan defaulters.
        """
    )


# --------------------------------------------------
# Input Form
# --------------------------------------------------

st.subheader("📝 Applicant Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
    )

    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=850.0,
        value=650.0,
        step=1.0,
    )

    months_employed = st.number_input(
        "Months Employed",
        min_value=0,
        value=60,
        step=1,
    )

    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=0,
        value=3,
        step=1,
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.0,
        step=0.1,
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=36,
        step=1,
    )


with col2:

    dti_ratio = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=0.30,
        step=0.01,
        format="%.2f",
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

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Employed",
            "Self-employed",
            "Unemployed",
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

    has_mortgage = st.selectbox(
        "Has Mortgage",
        [
            "Yes",
            "No",
        ],
    )

    has_dependents = st.selectbox(
        "Has Dependents",
        [
            "Yes",
            "No",
        ],
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Home",
            "Auto",
            "Education",
            "Business",
            "Other",
        ],
    )

    has_cosigner = st.selectbox(
        "Has Co-Signer",
        [
            "Yes",
            "No",
        ],
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Loan Default Risk",
    use_container_width=True,
):

    input_data = pd.DataFrame(
        [
            {
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
                "HasCoSigner": has_cosigner,
            }
        ]
    )

    try:

        # Get default probability
        probability = model.predict_proba(input_data)[0][1]

        # Apply business threshold
        prediction = int(probability >= THRESHOLD)

        # Determine risk level
        if probability >= 0.50:
            risk_level = "Very High Risk"
        elif probability >= THRESHOLD:
            risk_level = "High Risk"
        else:
            risk_level = "Low Risk"

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------

        st.subheader("📊 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Default Probability",
                f"{probability:.2%}",
            )

        with result_col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.0%}",
            )

        with result_col3:
            st.metric(
                "Prediction",
                "Default" if prediction == 1 else "No Default",
            )

        st.divider()

        if risk_level == "Very High Risk":

            st.error(
                f"🚨 **{risk_level}**\n\n"
                f"The applicant has a relatively high estimated "
                f"probability of default ({probability:.2%})."
            )

        elif risk_level == "High Risk":

            st.warning(
                f"⚠️ **{risk_level}**\n\n"
                f"The applicant's estimated default probability "
                f"is {probability:.2%}, which is above the "
                f"business threshold of {THRESHOLD:.0%}."
            )

        else:

            st.success(
                f"✅ **{risk_level}**\n\n"
                f"The applicant's estimated default probability "
                f"is {probability:.2%}, which is below the "
                f"business threshold of {THRESHOLD:.0%}."
            )

        # --------------------------------------------------
        # Business Interpretation
        # --------------------------------------------------

        st.subheader("💡 Business Interpretation")

        if prediction == 1:

            st.write(
                """
                The model recommends flagging this applicant
                for additional credit-risk review.

                Possible actions could include:

                - Additional financial verification
                - Manual underwriting review
                - Lower approved loan amount
                - Additional collateral or co-signer requirements
                """
            )

        else:

            st.write(
                """
                The applicant is currently classified as lower risk
                according to the model and selected decision threshold.

                The application may proceed to the next stage of
                the lending process, subject to other business rules.
                """
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Loan Default Prediction System | "
    "Machine Learning + MLOps Project"
)