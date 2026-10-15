from pathlib import Path

import pytest

import src.models.artifacts as artifacts_module


# Test that artifacts can be saved and loaded successfully.
def test_save_and_load_artifacts(tmp_path):
    artifacts = artifacts_module.ModelArtifacts(
        model={"name": "test_model"},
        preprocessor={"name": "test_preprocessor"},
        threshold=0.5,
        metadata={"model_version": "1.0"},
    )

    artifact_path = tmp_path / "test_artifacts.joblib"

    artifacts_module.save_artifacts(artifacts, artifact_path)
    loaded_artifacts = artifacts_module.load_artifacts(artifact_path)

    assert loaded_artifacts.model == artifacts.model
    assert loaded_artifacts.preprocessor == artifacts.preprocessor
    assert loaded_artifacts.threshold == artifacts.threshold
    assert loaded_artifacts.metadata == artifacts.metadata


# Test that a loading error is logged and re-raised.
def test_load_artifacts_logs_and_reraises_error(monkeypatch, caplog):
    import logging

    def fail_loading(path):
        raise OSError("Cannot load artifact")

    monkeypatch.setattr(artifacts_module.joblib, "load", fail_loading)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError, match="Cannot load artifact"):
            artifacts_module.load_artifacts(Path("missing.joblib"))

    assert "Failed to load model artifacts" in caplog.text