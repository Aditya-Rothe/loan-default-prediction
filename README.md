# 💰 Loan Default Prediction — End-to-End MLOps Project

An end-to-end **Loan Default Prediction system** built using Machine Learning and modern MLOps practices.

The project covers the complete ML lifecycle — from data ingestion and preprocessing to model training, experiment tracking, data/model versioning, automated testing, containerization, API deployment, and a Streamlit prediction application.

---

## 🚀 Live Application

**Streamlit App:**
Add your deployed Streamlit Community Cloud URL here.

> Example: https://loan-default-prediction-4aivfdbdf3d4swmbe5dhh6.streamlit.app/

---

## 📌 Project Overview

Loan default prediction is a classification problem where the objective is to identify applicants who are likely to default on their loans.

In lending businesses, correctly identifying high-risk applicants is important because:

* Loan defaults can cause financial losses.
* Manual risk assessment can be time-consuming.
* Missing potential defaulters can be costly.
* A machine learning system can provide consistent risk predictions.

This project builds a machine learning pipeline that predicts the probability of loan default and classifies applicants into different risk levels.

---

## 🎯 Business Objective

The primary objective is to develop a machine learning system that can:

1. Predict the probability of loan default.
2. Identify high-risk applicants.
3. Improve the detection of potential defaulters.
4. Provide an API for model predictions.
5. Provide an interactive web application.
6. Track experiments and model performance.
7. Version datasets and models.
8. Automate testing using CI/CD practices.

---

# 📊 Dataset

The project uses a loan default dataset containing **255,347 records and 18 original features**.

### Original Features

| Feature        | Description                       |
| -------------- | --------------------------------- |
| LoanID         | Unique loan identifier            |
| Age            | Applicant age                     |
| Income         | Annual income                     |
| LoanAmount     | Loan amount                       |
| CreditScore    | Applicant credit score            |
| MonthsEmployed | Employment duration               |
| NumCreditLines | Number of credit lines            |
| InterestRate   | Loan interest rate                |
| LoanTerm       | Loan duration                     |
| DTIRatio       | Debt-to-income ratio              |
| Education      | Applicant education               |
| EmploymentType | Employment category               |
| MaritalStatus  | Marital status                    |
| HasMortgage    | Whether applicant has a mortgage  |
| HasDependents  | Whether applicant has dependents  |
| LoanPurpose    | Purpose of the loan               |
| HasCoSigner    | Whether applicant has a co-signer |
| Default        | Target variable                   |

---

# 🏗️ Project Architecture

```text
                         ┌─────────────────────┐
                         │     Loan Dataset    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Ingestion    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Preprocessing  │
                         │                     │
                         │ • Scaling           │
                         │ • Encoding          │
                         │ • Feature Selection │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Training      │
                         │                     │
                         │ Gradient Boosting   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Evaluation       │
                         │                     │
                         │ ROC-AUC / F1 /      │
                         │ Precision / Recall  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          ┌──────────────────┐             ┌──────────────────┐
          │     MLflow       │             │       DVC        │
          │ Experiment       │             │ Data & Model     │
          │ Tracking         │             │ Versioning       │
          └──────────────────┘             └────────┬─────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                           │     DAGsHub      │
                                           │ Remote Storage   │
                                           └──────────────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                           │  Streamlit Cloud │
                                           │   Deployment     │
                                           └────────┬─────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                           │ Loan Risk        │
                                           │ Prediction App   │
                                           └──────────────────┘
```

---

# 🔬 Machine Learning Pipeline

## 1. Data Ingestion

The raw dataset is loaded from:

```text
data/raw/Loan_default.csv
```

The ingestion script validates and stores the processed dataset in:

```text
data/processed/loan_default.csv
```

---

## 2. Data Preprocessing

The preprocessing pipeline performs:

### Numerical Features

* Age
* Income
* LoanAmount
* CreditScore
* MonthsEmployed
* NumCreditLines
* InterestRate
* LoanTerm
* DTIRatio

Numerical features are standardized using:

```text
StandardScaler
```

### Categorical Features

* Education
* EmploymentType
* MaritalStatus
* HasMortgage
* HasDependents
* LoanPurpose
* HasCoSigner

Categorical features are transformed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

The final transformed feature space contains **31 features**.

---

# 🤖 Model

The selected model is:

**Gradient Boosting Classifier**

The model is implemented inside a Scikit-learn Pipeline containing the preprocessing and classification steps.

The trained pipeline is saved as:

```text
models/loan_default_pipeline.pkl
```

The model file is intentionally **not stored directly in GitHub**.

Instead, it is versioned using **DVC** and stored remotely on **DAGsHub**.

---

# 📈 Model Evaluation

The model was evaluated using multiple classification metrics.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  |     0.8324 |
| Precision |     0.3272 |
| Recall    |     0.4197 |
| F1 Score  |     0.3677 |
| ROC-AUC   | **0.7580** |

### Why ROC-AUC?

Because loan default prediction is a classification problem where identifying risky applicants is important, ROC-AUC provides a useful measure of how well the model separates defaulting and non-defaulting applicants across different probability thresholds.

---

# 🎯 Business Threshold Optimization

The default Scikit-learn classification threshold is:

```text
0.50
```

However, using 0.50 resulted in very low recall.

### Threshold Comparison

| Threshold |  Precision |     Recall |         F1 |
| --------: | ---------: | ---------: | ---------: |
|      0.50 |     0.6261 |     0.0494 |     0.0916 |
|      0.40 |     0.5383 |     0.1101 |     0.1828 |
|      0.30 |     0.4294 |     0.2210 |     0.2919 |
|      0.25 |     0.3787 |     0.3062 |     0.3386 |
|  **0.20** | **0.3272** | **0.4197** | **0.3677** |
|      0.15 |     0.2719 |     0.5655 |     0.3672 |
|      0.10 |     0.2087 |     0.7469 |     0.3262 |

A threshold of:

```text
0.20
```

was selected as the business decision threshold.

This increases the model's ability to identify potential defaulters compared with the default 0.50 threshold.

> The threshold is a business decision and should ultimately be tuned according to the financial cost of false positives versus false negatives.

---

# ⚙️ MLOps Implementation

This project implements several MLOps practices.

## 🔹 Experiment Tracking — MLflow

MLflow is used to track:

* Model parameters
* Evaluation metrics
* Experiments
* Model training runs

---

## 🔹 Data & Model Versioning — DVC

DVC is used to version large data/model artifacts that should not be stored directly in Git.

The model is stored remotely using:

```text
DAGsHub
```

The model can be retrieved using:

```bash
dvc pull -r origin models/loan_default_pipeline.pkl
```

---

## 🔹 CI — GitHub Actions

GitHub Actions automatically runs the test suite when code is pushed or a pull request is created.

Pipeline:

```text
Git Push / Pull Request
        ↓
GitHub Actions
        ↓
Install Dependencies
        ↓
Retrieve DVC Model
        ↓
Run Pytest
        ↓
Pass / Fail
```

---

## 🔹 Automated Testing

The API contains automated tests for:

* Root endpoint
* Health endpoint
* Prediction endpoint

Current test result:

```text
3 tests passed
```

Tests can be executed using:

```bash
python -m pytest
```

---

# 🐳 Docker

The project includes a Docker configuration for containerizing the prediction API.

Build the image:

```bash
docker build -t loan-default-api .
```

Run the container:

```bash
docker run -p 8000:8000 loan-default-api
```

The API will then be available at:

```text
http://localhost:8000
```

---

# 🚀 FastAPI

The trained model is exposed through a REST API using FastAPI.

### Available Endpoints

#### Health Check

```http
GET /
```

Returns:

```json
{
  "message": "Loan Default Prediction API is running"
}
```

---

#### API Health

```http
GET /health
```

Returns:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

#### Prediction

```http
POST /predict
```

The endpoint accepts applicant information and returns:

* Default probability
* Prediction
* Risk level
* Decision threshold

Example response:

```json
{
  "default_probability": 0.2745,
  "prediction": 1,
  "risk_level": "High Risk",
  "threshold": 0.2
}
```

---

# 🌐 Streamlit Application

The project also includes an interactive Streamlit application.

The application allows users to enter applicant information and receive:

```text
Default Probability
        ↓
Business Threshold
        ↓
Prediction
        ↓
Risk Level
```

### Risk Classification

|   Probability | Risk Level     |
| ------------: | -------------- |
|        < 0.20 | Low Risk       |
| 0.20 – < 0.50 | High Risk      |
|        ≥ 0.50 | Very High Risk |

---

# ☁️ Streamlit Cloud + DVC + DAGsHub

The Streamlit application does not require the model file to be committed to GitHub.

During deployment:

```text
Streamlit Cloud
       ↓
Check model
       ↓
Model not found
       ↓
Read DAGsHub credentials
from Streamlit Secrets
       ↓
Configure DVC authentication
       ↓
DVC Pull
       ↓
DAGsHub
       ↓
Download model
       ↓
Load model
       ↓
Start application
```

DAGsHub credentials are stored securely using Streamlit Secrets and are **not hardcoded in the source code**.

---

# 📁 Project Structure

```text
loan-default-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   │   └── Loan_default.csv
│   │
│   └── processed/
│       └── loan_default.csv
│
├── models/
│   └── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   └── test_api.py
│
├── app.py
├── setup_dvc.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Gradient Boosting Classifier
* Classification Metrics
* Probability Threshold Optimization

### MLOps

* MLflow
* DVC
* DAGsHub
* Git
* GitHub
* GitHub Actions

### Deployment

* FastAPI
* Uvicorn
* Streamlit
* Streamlit Community Cloud
* Docker

### Testing

* Pytest

---

# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Aditya-Rothe/loan-default-prediction.git
```

```bash
cd loan-default-prediction
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Retrieve DVC Model

Configure your DAGsHub credentials and pull the model:

```bash
dvc pull -r origin models/loan_default_pipeline.pkl
```

---

# 🧪 Run Tests

```bash
python -m pytest
```

Expected result:

```text
3 passed
```

---

# 🌐 Run Streamlit

```bash
streamlit run app.py
```

---

# 🚀 Run FastAPI

```bash
uvicorn src.api:app --reload
```

API documentation will be available through FastAPI's interactive documentation.

---

# 🐳 Run with Docker

Build:

```bash
docker build -t loan-default-api .
```

Run:

```bash
docker run -p 8000:8000 loan-default-api
```

---

# 🔄 Complete MLOps Workflow

The complete workflow implemented in this project is:

```text
Business Understanding
        ↓
Data Ingestion
        ↓
Data Validation / Processing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Threshold Optimization
        ↓
MLflow Experiment Tracking
        ↓
DVC Data / Model Versioning
        ↓
DAGsHub Remote Storage
        ↓
Pytest
        ↓
GitHub Actions CI
        ↓
Docker
        ↓
FastAPI
        ↓
Streamlit
        ↓
Streamlit Cloud
```

---

# 📌 Key Learning Outcomes

Through this project, I implemented:

* End-to-end machine learning workflow
* Data preprocessing pipelines
* Classification model development
* Model evaluation
* Business-oriented threshold optimization
* Experiment tracking with MLflow
* Data and model versioning with DVC
* Remote artifact storage with DAGsHub
* Automated testing with Pytest
* CI using GitHub Actions
* REST API development with FastAPI
* Docker containerization
* Streamlit application development
* Cloud deployment
* Secure credential management
* Reproducible ML workflows

---

# 🔮 Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Model explainability using SHAP
* Data drift detection
* Model monitoring
* Automated model retraining
* Performance monitoring
* Feature importance dashboard
* Better handling of class imbalance
* Model registry and production promotion workflow
* Cloud-based ML monitoring

---

# 👨‍💻 Author

**Aditya Rothe**

B.Sc. Data Science

Interested in:

* Data Science
* Machine Learning
* Machine Learning Engineering
* MLOps

---

## ⭐ If you found this project useful

Feel free to explore the repository, experiment with the pipeline, and build upon it.
