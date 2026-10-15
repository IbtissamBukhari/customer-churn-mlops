
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from src.features.preprocessing import prepare_target

from src.features.preprocessing import (
    clean_total_charges,
    prepare_target,
    prepare_features,
    create_preprocessor
)

def test_prepare_target_converts_churn_labels():
    y = pd.Series(["No", "Yes", "No"])

    result = prepare_target(y)

    assert result.tolist() == [0, 1, 0]


def test_prepare_target_rejects_missing_values():
    y = pd.Series(["No", None, "Yes"])

    with pytest.raises(ValueError, match="missing values"):
        prepare_target(y)


def test_prepare_target_rejects_unexpected_values():
    y = pd.Series(["No", "Maybe", "Yes"])

    with pytest.raises(ValueError, match="Unexpected target values"):
        prepare_target(y)


def test_clean_total_charges_fills_blank_for_zero_tenure():
    df = pd.DataFrame({
        "tenure": [0],
        "TotalCharges": [" "],
    })

    result = clean_total_charges(df)

    assert result["TotalCharges"].iloc[0] == 0.0


def test_clean_total_charges_rejects_invalid_text():
    df = pd.DataFrame({
        "tenure": [12],
        "TotalCharges": ["invalid"],
    })

    with pytest.raises(ValueError, match="TotalCharges"):
        clean_total_charges(df)



def test_clean_total_charges_rejects_missing_charges_for_positive_tenure():
    df = pd.DataFrame({
        "tenure": [12],
        "TotalCharges": [None],
    })

    with pytest.raises(ValueError, match="TotalCharges"):
        clean_total_charges(df)


def test_prepare_features_removes_identifier_and_target():
    df = pd.DataFrame({
        "customerID": ["001"],
        "tenure": [12],
        "MonthlyCharges": [50.0],
        "TotalCharges": ["600"],
        "Churn": ["No"],
    })

    result = prepare_features(df)

    assert "customerID" not in result.columns
    assert "Churn" not in result.columns
    assert result["TotalCharges"].iloc[0] == 600.0


def test_prepare_features_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        prepare_features(df)


def test_create_preprocessor_returns_column_transformer():
    preprocessor = create_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = {
        name for name, _, _ in preprocessor.transformers
    }

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names
