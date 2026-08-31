from fastapi.testclient import TestClient

from src.api import app


def test_root():

    with TestClient(app) as client:

        response = client.get("/")

        assert response.status_code == 200


def test_health():

    with TestClient(app) as client:

        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_prediction():

    payload = {
        "Age": 35,
        "Income": 50000,
        "LoanAmount": 150000,
        "CreditScore": 650,
        "MonthsEmployed": 60,
        "NumCreditLines": 3,
        "InterestRate": 10.5,
        "LoanTerm": 36,
        "DTIRatio": 0.35,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "No",
        "LoanPurpose": "Home",
        "HasCoSigner": "Yes",
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        data = response.json()

        assert "default_probability" in data
        assert "prediction" in data
        assert "risk_level" in data
        assert "threshold" in data

        assert 0 <= data["default_probability"] <= 1
        assert data["prediction"] in [0, 1]
