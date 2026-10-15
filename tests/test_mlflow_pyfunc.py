import logging

import pandas as pd
import pytest

import src.models.mlflow_pyfunc as pyfunc_module


# Test that load_context loads the churn artifact successfully.
def test_load_context_success(monkeypatch):
    fake_artifact = {"model": "fake_model"}

    monkeypatch.setattr(
        pyfunc_module.joblib,
        "load",
        lambda path: fake_artifact,
    )

    model = pyfunc_module.ChurnPredictionModel()

    class FakeContext:
        artifacts = {"churn_artifact": "fake_artifact.joblib"}

    model.load_context(FakeContext())

    assert model.artifact == fake_artifact


# Test that a prediction failure is logged and re-raised.
def test_predict_logs_and_reraises_error(monkeypatch, caplog):
    model = pyfunc_module.ChurnPredictionModel()
    model.artifact = object()

    def fail_prediction(customer_data, artifact):
        raise RuntimeError("Customer prediction failed")

    monkeypatch.setattr(
        pyfunc_module,
        "predict_customer",
        fail_prediction,
    )

    model_input = pd.DataFrame({"tenure": [12]})

    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            RuntimeError,
            match="Customer prediction failed",
        ):
            model.predict(None, model_input)

    assert "Churn prediction failed." in caplog.text