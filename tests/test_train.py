import logging

import numpy as np
import pytest

import src.models.train as train_module


# Test that train_model fits and returns an XGBoost model.
def test_train_model_success(monkeypatch):
    class FakeModel:
        def fit(self, X, y):
            self.fitted = True

    monkeypatch.setattr(
        train_module,
        "XGBClassifier",
        lambda **kwargs: FakeModel(),
    )

    model = train_module.train_model(
        np.array([[1], [2]]),
        np.array([0, 1]),
    )

    assert model.fitted is True


# Test that a training error is logged and re-raised.
def test_train_model_logs_and_reraises_error(caplog, monkeypatch):
    def fail_model_creation(**kwargs):
        raise RuntimeError("Model training failed")

    monkeypatch.setattr(
        train_module,
        "XGBClassifier",
        fail_model_creation,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Model training failed"):
            train_module.train_model(
                np.array([[1], [2]]),
                np.array([0, 1]),
            )

    assert "XGBoost model training failed." in caplog.text