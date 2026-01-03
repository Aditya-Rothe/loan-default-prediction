# Loan Default Prediction | Machine Learning Project

### Overview
Loan default prediction is a critical problem in the banking and financial sector.
This project focuses on building a machine learning–based classification system to predict whether a loan applicant is likely to default on a loan based on their financial, personal, and loan-related attributes.

The project follows a complete end-to-end data science workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and comparison of multiple ML algorithms.

### Objective

- The main objectives of this project are:
- To analyze customer loan data and identify key risk factors
-  To build and evaluate multiple machine learning models for loan default prediction
- To compare models using appropriate metrics for imbalanced datasets
- To gain practical experience with real-world financial data

### Dataset Description

- Source: Kaggle (Loan Default Dataset)
- Records: ~255,000 loan applications
- Features: 18 input variables + 1 target variable

Target Variable:
Default
0 → No Default
1 → Loan Default

### Feature Types

Numerical Features:
Age, Income, LoanAmount, CreditScore, InterestRate, DTIRatio, etc.

Categorical Features:
Education, EmploymentType, MaritalStatus, LoanPurpose, HasMortgage, HasCoSigner, etc.

### Project Workflow

The project was executed in the following structured steps:

### 1. Data Understanding & Cleaning
- Dataset inspection (head, shape, info)
- Feature type classification (numerical vs categorical)
- Removal of irrelevant identifiers (LoanID)

### 2. Exploratory Data Analysis (EDA)

- Univariate Analysis: Distribution of numerical features
- Bivariate Analysis: Relationship between features and loan default
- Categorical Analysis: Default rate across categories
- Multivariate Analysis: Correlation heatmap for numerical features

### 3. Data Preprocessing

- Encoding of categorical variables
- Feature scaling using standardization
- Train–test split to avoid data leakage

### 4. Model Building

The following models were implemented and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

### 5. Model Evaluation

Models were evaluated using metrics suitable for imbalanced data, including:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC–AUC Score
- Confusion Matrix

### Key Insights

The dataset is highly imbalanced, making accuracy alone a misleading metric
ROC–AUC is more reliable for evaluating model performance
Income, Credit Score, Interest Rate, and DTI Ratio play a major role in loan default risk
Ensemble models outperform basic classifiers

### Future Improvements

- Apply resampling techniques (SMOTE, class weighting)
- Perform hyperparameter tuning (GridSearchCV)
- Improve recall by adjusting classification thresholds
- Use advanced models like XGBoost or LightGBM
- Deploy the model as a web application

### Skills Demonstrated

- Data Cleaning & Feature Engineering
- Exploratory Data Analysis (EDA)
- Handling Imbalanced Datasets
- Machine Learning Model Evaluation
- Python, Pandas, NumPy, Scikit-learn
- End-to-End ML Project Workflow
