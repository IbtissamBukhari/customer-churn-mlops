import pandas as pd
import pytest

from src.data.splitting import split_data
from src.exceptions import DataValidationError

def test_split_data_returns_six_outputs():
    X = pd.DataFrame({
        "tenure": range(100),
        "MonthlyCharges": range(100, 200),
    })

    y = pd.Series([0, 1] * 50)

    result = split_data(X, y)

    assert len(result) == 6


def test_split_data_uses_correct_proportions():
    X = pd.DataFrame({
        "tenure": range(1000),
        "MonthlyCharges": range(1000, 2000),
    })

    y = pd.Series([0, 1] * 500)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150

    assert len(y_train) == 700
    assert len(y_val) == 150
    assert len(y_test) == 150

def test_split_data_preserves_target_proportions():
    X = pd.DataFrame({
        "tenure": range(1000),
        "MonthlyCharges": range(1000, 2000),
    })

    y = pd.Series([0] * 700 + [1] * 300)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    original_churn_rate = y.mean()

    assert abs(y_train.mean() - original_churn_rate) < 0.02
    assert abs(y_val.mean() - original_churn_rate) < 0.02
    assert abs(y_test.mean() - original_churn_rate) < 0.02

# Test that data splitting rejects an empty feature DataFrame.
def test_split_data_rejects_empty_features():
    X = pd.DataFrame(columns=["tenure", "MonthlyCharges"])
    y = pd.Series([], dtype=int)

    with pytest.raises(DataValidationError, match="must not be empty"):
        split_data(X, y)

# Test that data splitting rejects mismatched feature and target row counts.
def test_split_data_rejects_mismatched_rows():
    X = pd.DataFrame({
        "tenure": range(10),
        "MonthlyCharges": range(10, 20),
    })

    y = pd.Series([0, 1] * 4)

    with pytest.raises(
        DataValidationError,
        match="same number of rows",
    ):
        split_data(X, y)