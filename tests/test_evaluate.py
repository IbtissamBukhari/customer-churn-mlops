import logging

import numpy as np
import pytest

import src.models.evaluate as evaluate


# Test that prediction probabilities are generated successfully.
def test_predict_probabilities_success():
    class FakeModel:
        def predict_proba(self, X):
            return np.array([
                [0.8, 0.2],
                [0.3, 0.7],
            ])

    result = evaluate.predict_probabilities(
        FakeModel(),
        np.array([[1], [2]]),
    )

    np.testing.assert_array_equal(result, np.array([0.2, 0.7]))


# Test that a probability prediction error is logged and re-raised.
def test_predict_probabilities_logs_and_reraises_error(caplog):
    class FakeModel:
        def predict_proba(self, X):
            raise RuntimeError("Prediction failed")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Prediction failed"):
            evaluate.predict_probabilities(
                FakeModel(),
                np.array([[1]]),
            )

    assert "Failed to generate prediction probabilities." in caplog.text