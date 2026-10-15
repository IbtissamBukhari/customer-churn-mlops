import pandas as pd
import pytest

from src.data.validation import (
    DataValidationError,
    validate_columns,
    validate_target_column,
    validate_numeric_ranges,
    validate_raw_data,
)
def test_validate_raw_data():
    df = pd.DataFrame({
        "customerID": ["001"],
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["Yes"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["One year"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [50.0],
        "TotalCharges": ["600"],
        "Churn": ["No"],
    })

    validate_raw_data(df)

def test_validate_target_column_rejects_invalid_value():
    df = pd.DataFrame({
        "Churn": ["Yes", "No", "Maybe"]
    })

    with pytest.raises(DataValidationError):
        validate_target_column(df)

def test_validate_numeric_ranges_rejects_negative_tenure():
    df = pd.DataFrame({
        "tenure": [12, -1],
        "MonthlyCharges": [50.0, 60.0],
    })

    with pytest.raises(DataValidationError):
        validate_numeric_ranges(df)

def test_validate_numeric_ranges_rejects_negative_monthly_charges():
    df = pd.DataFrame({
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, -10.0],
    })

    with pytest.raises(DataValidationError):
        validate_numeric_ranges(df)

def test_validate_columns_rejects_missing_column():
    df = pd.DataFrame({
        "customerID": ["001"],
        "gender": ["Female"],
    })

    with pytest.raises(DataValidationError):
        validate_columns(df)


def test_validate_numeric_ranges_rejects_infinite_tenure():
    df = pd.DataFrame({
        "tenure": [12, float("inf")],
        "MonthlyCharges": [50.0, 60.0],
    })

    with pytest.raises(DataValidationError):
        validate_numeric_ranges(df)


def test_validate_numeric_ranges_rejects_missing_values():
    df = pd.DataFrame({
        "tenure": [12, None],
        "MonthlyCharges": [50.0, 60.0],
    })

    with pytest.raises(DataValidationError):
        validate_numeric_ranges(df)


def test_validate_columns_rejects_unexpected_column():
    df = pd.DataFrame({
        "customerID": ["001"],
        "unexpected_column": ["value"],
    })

    with pytest.raises(DataValidationError):
        validate_columns(df)


def test_validate_numeric_ranges_rejects_non_numeric_tenure():
    df = pd.DataFrame({
        "tenure": [12, "unknown"],
        "MonthlyCharges": [50.0, 60.0],
    })

    with pytest.raises(DataValidationError):
        validate_numeric_ranges(df)

