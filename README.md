# Loan Default Prediction System

This project predicts whether a customer is likely to default on a loan using machine learning models.  
The final model is deployed using **Streamlit** for real-time prediction.

---

## 🚀 Project Features
- Predicts loan default risk (Yes / No)
- Uses Gradient Boosting Classifier
- Handles real-world imbalanced dataset
- Simple and interactive web interface
- Free deployment using Streamlit Cloud

---

## 📊 Machine Learning Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (Final Model)

Model comparison was done using:
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 🧠 Why ROC-AUC?
The dataset is imbalanced, so ROC-AUC was preferred over accuracy as the primary evaluation metric.

---

## 🖥️ Web App Input Features
- Age
- Income
- Loan Amount
- Credit Score
- Months Employed
- Number of Credit Lines
- Interest Rate
- Loan Term
- Debt-to-Income Ratio

---

## ▶️ How to Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
