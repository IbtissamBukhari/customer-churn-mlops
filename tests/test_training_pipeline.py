import logging

import pytest

import src.training_pipeline as pipeline


# Test that a failure during data loading is logged and re-raised.
def test_pipeline_logs_and_reraises_loading_error(caplog, monkeypatch):
    def fail_loading():
        raise RuntimeError("Data loading failed")

    monkeypatch.setattr(pipeline, "load_raw_data", fail_loading)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Data loading failed"):
            pipeline.run_training_pipeline()

    assert "Customer churn training pipeline failed." in caplog.text


# Test that the pipeline returns the expected results after successful execution.
def test_pipeline_returns_results(monkeypatch):
    expected_result = {
        "artifact_path": pipeline.Path("models/churn_model.joblib"),
        "validation_metrics": {"f1": 0.80},
        "test_metrics": {"f1": 0.78},
    }

    # We will mock the pipeline stages in the next step to simulate success.
    # This test is intentionally not complete yet.

class FakePreprocessor:
    def fit(self, X):
        return self

    def transform(self, X):
        return X

# Test that the pipeline returns metrics and the artifact path after success.
def test_pipeline_returns_results(monkeypatch):
    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        pipeline.mlflow, "start_run", lambda: FakeRun()
    )
    monkeypatch.setattr(pipeline.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(pipeline.mlflow, "log_param", lambda key, value: None)
    monkeypatch.setattr(pipeline.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(pipeline.mlflow, "log_artifact", lambda path: None)

    monkeypatch.setattr(pipeline, "load_raw_data", lambda: "fake_data")
    monkeypatch.setattr(pipeline, "validate_raw_data", lambda df: None)
    monkeypatch.setattr(pipeline, "prepare_features", lambda df: "fake_features")
    monkeypatch.setattr(pipeline, "prepare_target", lambda target: "fake_target")
    monkeypatch.setattr(
        pipeline, "split_data",
        lambda X, y: ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"),
    )
    monkeypatch.setattr(pipeline, "create_preprocessor", lambda: object())
    monkeypatch.setattr(pipeline, "train_model", lambda X, y: "fake_model")
    monkeypatch.setattr(pipeline, "predict_probabilities", lambda model, X: [0.8])
    monkeypatch.setattr(pipeline, "apply_threshold", lambda probs, threshold: [1])
    monkeypatch.setattr(
        pipeline, "calculate_metrics",
        lambda y, predictions, probabilities: {"f1": 0.80},
    )
    monkeypatch.setattr(pipeline, "save_artifacts", lambda artifact, path: None)

    # Use arrays that support the pipeline's expected conversion methods.
    import numpy as np
    import pandas as pd

    monkeypatch.setattr(
        pipeline, "load_raw_data",
        lambda: pd.DataFrame({pipeline.TARGET_COLUMN: ["Yes"]}),
    )
    monkeypatch.setattr(
        pipeline, "prepare_features",
        lambda df: pd.DataFrame({"feature": [1]}),
    )
    monkeypatch.setattr(
        pipeline, "prepare_target",
        lambda target: pd.Series([1]),
    )
    monkeypatch.setattr(
        pipeline, "split_data",
        lambda X, y: (
            X, X, X,
            pd.Series([1]), pd.Series([1]), pd.Series([1]),
        ),
    )
    monkeypatch.setattr(
        pipeline, "create_preprocessor",
        lambda: FakePreprocessor(),
    )

    result = pipeline.run_training_pipeline()

    assert "artifact_path" in result
    assert result["validation_metrics"] == {"f1": 0.80}
    assert result["test_metrics"] == {"f1": 0.80}