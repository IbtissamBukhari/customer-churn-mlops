from fastapi.testclient import TestClient
from app.main import app
import numpy as np
from src.models.predict import apply_threshold
import pytest
from unittest.mock import patch
import runpy
import pandas as pd

client=TestClient(app)

# Returns a fresh, valid customer dictionary for API tests.
def get_valid_customer():
    return {
        "customerID": "TEST001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 86.0,
        "TotalCharges": 1032.0,
    }

# Checks that the health endpoint confirms the API is alive.
def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

# Checks that the readiness endpoint confirms the model is available.
def test_readiness_check():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "available",
    }

# Checks that readiness fails when the model is unavailable.
def test_readiness_check_model_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.model", None)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Model is not available."
    }

# Checks that the prediction endpoint accepts valid customer data.
def test_predict_valid_customer():
    customer = get_valid_customer()
    response=client.post("/predict", json=customer)
    assert response.status_code==200
    result = response.json()
    assert "churn_probability" in result
    assert "prediction" in result
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert isinstance(result["churn_probability"], (int, float))
    assert isinstance(result["prediction"], int)
    assert result["prediction"] == int(result["churn_probability"] >= 0.30)

# Checks that the API rejects an invalid SeniorCitizen value.
def test_predict_invalid_senior_citizen():
    customer=get_valid_customer()
    customer["SeniorCitizen"] = 3  # Invalid value
    response=client.post("/predict", json=customer)
    assert response.status_code==422  # Unprocessable Entity

# Checks that the API rejects negative customer tenure.
def test_predict_invalid_tenure():
    customer = get_valid_customer()
    customer["tenure"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

# Checks that inconsistent phone service data is rejected.
def test_predict_invalid_phone_service_combination():
    customer = get_valid_customer()
    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "Yes"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

# Checks that the API rejects requests missing a required field.
def test_predict_missing_required_field():
    customer = get_valid_customer()
    customer.pop("tenure")

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

# Checks that unsupported categorical values are rejected.
def test_predict_invalid_gender():
    customer = get_valid_customer()
    customer["gender"] = "Unknown"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

# Checks that probabilities are converted to predictions using the threshold.
def test_apply_threshold():
    probabilities = np.array([0.10, 0.30, 0.80])

    predictions = apply_threshold(probabilities, threshold=0.30)

    assert predictions.tolist() == [0, 1, 1]


# Checks that an invalid threshold raises a ValueError.
def test_apply_threshold_invalid_value():
    probabilities = np.array([0.20, 0.80])

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        apply_threshold(probabilities, threshold=1.5)

# Checks that the minimum and maximum valid thresholds work.
def test_apply_threshold_valid_boundaries():
    probabilities = np.array([0.0, 0.5, 1.0])

    predictions_at_zero = apply_threshold(probabilities, threshold=0.0)
    predictions_at_one = apply_threshold(probabilities, threshold=1.0)

    assert predictions_at_zero.tolist() == [1, 1, 1]
    assert predictions_at_one.tolist() == [0, 0, 1]

# Checks that probabilities outside the valid range are rejected.
def test_apply_threshold_invalid_probability():
    probabilities = np.array([0.20, 1.5])

    with pytest.raises(
        ValueError,
        match="Probabilities must be between 0 and 1."
    ):
        apply_threshold(probabilities, threshold=0.30)

# Checks that NaN probabilities are rejected.
def test_apply_threshold_nan_probability():
    probabilities = np.array([0.20, np.nan])

    with pytest.raises(
        ValueError,
        match="Probabilities must be between 0 and 1."
    ):
        apply_threshold(probabilities, threshold=0.30)

# Checks that infinite probabilities are rejected.
def test_apply_threshold_infinite_probability():
    probabilities = np.array([0.20, np.inf, -np.inf])

    with pytest.raises(
        ValueError,
        match="Probabilities must be between 0 and 1."
    ):
        apply_threshold(probabilities, threshold=0.30)

# Checks that an empty probability array is rejected.
def test_apply_threshold_empty_probabilities():
    probabilities = np.array([])

    with pytest.raises(
        ValueError,
        match="Probabilities cannot be empty."
    ):
        apply_threshold(probabilities, threshold=0.30)

# Checks that unexpected model prediction errors return HTTP 500.
def test_predict_model_failure(monkeypatch):
    class FakeModel:
        def predict(self, customer_df):
            raise ValueError("Simulated model prediction failure")

    monkeypatch.setattr("app.main.model", FakeModel())

    customer = get_valid_customer()
    response = client.post("/predict", json=customer)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction failed due to an internal server error."
    }

# Checks that the API starts and logs the error when model loading fails.
def test_model_loading_failure():
    with patch("logging.basicConfig"):
        with patch(
            "mlflow.pyfunc.load_model",
            side_effect=RuntimeError("Simulated MLflow loading failure"),
        ):
            app_namespace = runpy.run_path("app/main.py")

    assert app_namespace["model"] is None
    assert "app" in app_namespace


# Checks that prediction is rejected when the model is unavailable.
def test_predict_model_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.model", None)

    customer = get_valid_customer()
    response = client.post("/predict", json=customer)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction service is temporarily unavailable."
    }

# Checks that invalid requests return a safe validation message.
def test_predict_invalid_request_safe_message():
    customer = get_valid_customer()
    customer["gender"] = "Unknown"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request data. Please check your input."
    }

# Checks that unexpected model output returns a safe error response.
def test_predict_unexpected_model_output(monkeypatch):
    class FakeModel:
        def predict(self, customer_df):
            return pd.DataFrame({"unexpected_column": [1]})

    monkeypatch.setattr("app.main.model", FakeModel())

    customer = get_valid_customer()
    response = client.post("/predict", json=customer)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction failed due to an internal server error."
    }

# Checks that invalid churn probabilities are rejected safely.
@pytest.mark.parametrize("probability", [1.5, float("nan")])
def test_predict_invalid_probability(monkeypatch, probability):
    class FakeModel:
        def predict(self, customer_df):
            return pd.DataFrame({
                "prediction": [1],
                "churn_probability": [probability],
            })

    monkeypatch.setattr("app.main.model", FakeModel())

    customer = get_valid_customer()
    response = client.post("/predict", json=customer)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction failed due to an internal server error."
    }

# Checks that FastAPI invokes the custom prediction exception handler.
def test_model_prediction_error_uses_custom_handler():
    from app.exceptions import ModelPredictionError

    @app.get("/test-model-prediction-error")
    def trigger_prediction_error():
        raise ModelPredictionError("Internal diagnostic details.")

    response = client.get("/test-model-prediction-error")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction failed due to an internal server error."
    }


# Checks that negative billing amounts are rejected.
@pytest.mark.parametrize(
    "field",
    ["MonthlyCharges", "TotalCharges"],
)
def test_predict_negative_charges(field):
    customer = get_valid_customer()
    customer[field] = -10.0

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request data. Please check your input."
    }


# Checks that unsupported contract and payment method values are rejected.
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("Contract", "Monthly"),
        ("PaymentMethod", "Crypto"),
    ],
)
def test_predict_invalid_categorical_values(field, invalid_value):
    customer = get_valid_customer()
    customer[field] = invalid_value

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request data. Please check your input."
    }
