# 🏦 Loan Default Prediction – End-to-End MLOps Project

An end-to-end Machine Learning and MLOps project for predicting the probability of loan default. The project implements a complete machine learning lifecycle, from data ingestion and validation to model training, experiment tracking, data versioning, API deployment, containerization, automated testing, and Continuous Integration.

---

## 📌 Business Problem

Financial institutions need to identify customers who are likely to default on their loans.

Incorrectly approving a high-risk loan can result in financial losses. Therefore, the objective of this project is to build a machine learning system that predicts the probability of loan default and helps identify potentially risky loan applications.

The project uses a **Gradient Boosting Classifier** to predict whether a customer is likely to default on a loan.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Build a machine learning model for loan default prediction.
* Create a reproducible machine learning pipeline.
* Track experiments using MLflow.
* Version datasets and pipeline artifacts using DVC.
* Store DVC artifacts remotely using DAGsHub.
* Expose predictions through a FastAPI REST API.
* Containerize the application using Docker.
* Write automated API tests using Pytest.
* Automate testing using GitHub Actions CI.

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Raw Loan Dataset  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Ingestion    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Validation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    │ Scaling + Encoding  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Training      │
                    │ Gradient Boosting   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌─────────────┐       ┌─────────────┐
             │   MLflow    │       │ Model File  │
             │ Experiments │       │   .pkl      │
             └─────────────┘       └──────┬──────┘
                                          │
                                          ▼
                                  ┌─────────────┐
                                  │   FastAPI   │
                                  └──────┬──────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │   Docker    │
                                  └──────┬──────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │ Deployment  │
                                  └─────────────┘


        DVC + DAGsHub → Data & Artifact Versioning

        GitHub Actions → Automated Testing (CI)
```

---

# 📂 Project Structure

```text
loan-default-prediction/
│
├── data/
│   ├── raw/
│   │   ├── Loan_default.csv.dvc
│   │   └── .gitignore
│   │
│   └── processed/
│
├── models/
│   └── loan_default_pipeline.pkl
│
├── notebooks/
│   └── loan_default_prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── threshold_analysis.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dvc/
│   └── config
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The dataset contains information related to loan applicants, including:

### Numerical Features

* Age
* Income
* Loan Amount
* Credit Score
* Months Employed
* Number of Credit Lines
* Interest Rate
* Loan Term
* Debt-to-Income Ratio

### Categorical Features

* Education
* Employment Type
* Marital Status
* Has Mortgage
* Has Dependents
* Loan Purpose
* Has Co-Signer

### Target Variable

```text
Default
```

Where:

* `0` → Customer did not default.
* `1` → Customer defaulted.

The dataset contains approximately:

```text
255,347 records
18 columns
```

---

# 🔄 Machine Learning Pipeline

The project follows a structured machine learning workflow.

## 1. Data Ingestion

The raw dataset is loaded from:

```text
data/raw/Loan_default.csv
```

The ingestion pipeline saves the processed dataset to:

```text
data/processed/loan_default.csv
```

---

## 2. Data Validation

The dataset is validated before training.

The validation process checks:

* Required columns.
* Missing values.
* Duplicate records.
* Target variable validity.

This helps detect data problems before they reach the machine learning model.

---

## 3. Data Preprocessing

The preprocessing pipeline uses Scikit-learn's `ColumnTransformer`.

### Numerical Features

Numerical features are transformed using:

```text
StandardScaler
```

### Categorical Features

Categorical features are transformed using:

```text
OneHotEncoder
```

The preprocessing pipeline is combined with the machine learning model to ensure consistent transformations during training and prediction.

---

# 🤖 Model Training

The project uses:

```text
GradientBoostingClassifier
```

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

The trained pipeline is saved as:

```text
models/loan_default_pipeline.pkl
```

---

# 📈 Model Evaluation

The model is evaluated using multiple classification metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Example results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8324 |
| Precision | 0.3272 |
| Recall    | 0.4197 |
| F1 Score  | 0.3677 |
| ROC-AUC   | 0.7580 |

---

# 🎯 Threshold Optimization

Loan default prediction is an imbalanced classification problem.

Using the default classification threshold of `0.50` resulted in very low recall for detecting loan defaults.

Therefore, multiple thresholds were analyzed.

The final business threshold selected was:

```text
0.20
```

This improves the model's ability to identify potentially risky customers.

The decision is based on the business objective of reducing the number of missed loan defaults.

---

# 🧪 MLflow Experiment Tracking

MLflow is used to track machine learning experiments.

The project logs:

### Parameters

* Model name
* Random state
* Test size
* Classification threshold

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

MLflow helps compare experiments and maintain reproducibility.

Run MLflow locally using:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 📦 DVC – Data Version Control

DVC is used for:

* Dataset versioning.
* Pipeline reproducibility.
* Model artifact tracking.

The raw dataset is tracked using:

```bash
dvc add data/raw/Loan_default.csv
```

The project pipeline is defined in:

```text
dvc.yaml
```

The pipeline includes:

```text
Data Ingestion
        ↓
Model Training
```

The pipeline can be reproduced using:

```bash
dvc repro
```

Pipeline status can be checked using:

```bash
dvc status
```

---

# ☁️ DAGsHub Integration

DAGsHub is used as remote storage for DVC artifacts.

The DVC remote stores versioned artifacts such as:

* Raw dataset.
* Processed dataset.
* Trained model.

Artifacts can be uploaded using:

```bash
dvc push
```

Artifacts can be downloaded using:

```bash
dvc pull
```

---

# 🚀 FastAPI

The trained model is exposed through a REST API using FastAPI.

## Start the API

```bash
uvicorn src.api:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

### Root Endpoint

```text
GET /
```

Returns a message confirming that the API is running.

---

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### Prediction Endpoint

```text
POST /predict
```

The API accepts loan applicant information and returns:

* Default probability.
* Prediction.
* Risk level.
* Classification threshold.

Example response:

```json
{
  "default_probability": 0.32,
  "prediction": 1,
  "risk_level": "High Risk",
  "threshold": 0.2
}
```

---

# 🐳 Docker

The FastAPI application is containerized using Docker.

## Build the Docker Image

```bash
docker build -t loan-default-api .
```

## Run the Container

```bash
docker run -p 8000:8000 loan-default-api
```

The API will be available at:

```text
http://localhost:8000
```

---

# 🧪 Automated Testing

API endpoints are tested using Pytest.

Tests include:

* Root endpoint test.
* Health endpoint test.
* Prediction endpoint test.

Run the tests using:

```bash
python -m pytest
```

Example output:

```text
3 passed
```

---

# ⚙️ GitHub Actions CI

GitHub Actions is used for Continuous Integration.

Whenever code is pushed to the `main` branch:

```text
Push to GitHub
        ↓
GitHub Actions Triggered
        ↓
Set Up Python Environment
        ↓
Install Dependencies
        ↓
Configure DVC
        ↓
Pull Versioned Artifacts
        ↓
Run Automated Tests
        ↓
CI Passed ✅
```

This ensures that new code changes are automatically tested.

---

# 🛠️ Technologies Used

| Technology     | Purpose                    |
| -------------- | -------------------------- |
| Python         | Programming Language       |
| Pandas         | Data Processing            |
| Scikit-learn   | Machine Learning           |
| MLflow         | Experiment Tracking        |
| DVC            | Data & Pipeline Versioning |
| DAGsHub        | Remote Storage for DVC     |
| FastAPI        | REST API                   |
| Docker         | Containerization           |
| Pytest         | Automated Testing          |
| GitHub Actions | Continuous Integration     |
| Git & GitHub   | Version Control            |

---

# ⚡ Installation

## Clone the Repository

```bash
git clone https://github.com/Aditya-Rothe/loan-default-prediction.git

cd loan-default-prediction
```

## Create a Virtual Environment

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

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Pull DVC Artifacts

```bash
dvc pull
```

## Run the Pipeline

```bash
dvc repro
```

## Run Tests

```bash
python -m pytest
```

## Start the API

```bash
uvicorn src.api:app --reload
```

---

# 🔮 Future Improvements

Potential future improvements include:

* Hyperparameter optimization.
* Model monitoring.
* Data drift detection.
* Automated model retraining.
* Cloud deployment.
* Kubernetes deployment.
* Model registry integration.
* Feature store integration.
* Airflow orchestration.

---

# 👨‍💻 Author

**Aditya Rothe**

Aspiring Data Scientist | Machine Learning Engineer

---

# ⭐ If You Like This Project

If you found this project interesting, consider giving the repository a ⭐.
