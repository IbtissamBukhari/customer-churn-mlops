import logging

import numpy as np
import pandas as pd
import pytest

import src.models.predict as predict_module


# Test that predict_customer returns a probability and prediction.
def test_predict_customer_success(monkeypatch):
    class FakePreprocessor:
        def transform(self, features):
            return np.array([[1, 2]])

    class FakeModel:
        def predict_proba(self, features):
            return np.array([[0.2, 0.8]])

    class FakeArtifact:
        preprocessor = FakePreprocessor()
        model = FakeModel()
        threshold = 0.5

    monkeypatch.setattr(
        predict_module,
        "prepare_features",
        lambda customer_data: customer_data,
    )

    result = predict_module.predict_customer(
        pd.DataFrame({"tenure": [12]}),
        FakeArtifact(),
    )

    assert result == {
        "churn_probability": 0.8,
        "prediction": 1,
    }


# Test that prediction errors are logged and re-raised.
def test_predict_customer_logs_and_reraises_error(caplog, monkeypatch):
    def fail_preparation(customer_data):
        raise ValueError("Feature preparation failed")

    monkeypatch.setattr(
        predict_module,
        "prepare_features",
        fail_preparation,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Feature preparation failed"):
            predict_module.predict_customer(
                pd.DataFrame({"tenure": [12]}),
                object(),
            )

    assert "Customer churn prediction failed." in caplog.text