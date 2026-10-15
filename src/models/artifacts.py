import logging
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import joblib
import logging_config


logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    model: Any
    preprocessor: Any
    threshold: float
    metadata: dict


def save_artifacts(artifacts: ModelArtifacts, path: Path) -> None:
    logger.info("Saving model artifacts to %s.", path)

    try:
        joblib.dump(artifacts, path)
        logger.info("Model artifacts saved successfully.")

    except Exception:
        logger.exception("Failed to save model artifacts to %s.", path)
        raise


def load_artifacts(path: Path) -> ModelArtifacts:
    logger.info("Loading model artifacts from %s.", path)

    try:
        artifacts = joblib.load(path)
        logger.info("Model artifacts loaded successfully.")
        return artifacts

    except Exception:
        logger.exception("Failed to load model artifacts from %s.", path)
        raise