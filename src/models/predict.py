import logging

import logging_config
import numpy as np

from src.features.preprocessing import prepare_features
from src.models.artifacts import ModelArtifacts


logger = logging.getLogger(__name__)


def apply_threshold(probabilities, threshold: float):
    logger.info("Applying prediction threshold.")

    try:
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        if probabilities.size == 0:
            raise ValueError("Probabilities cannot be empty.")

        if not (
            np.isfinite(probabilities)
            & (probabilities >= 0)
            & (probabilities <= 1)
        ).all():
            raise ValueError("Probabilities must be between 0 and 1.")

        return (probabilities >= threshold).astype(int)

    except Exception:
        logger.exception("Failed to apply prediction threshold.")
        raise


def predict_customer(
    customer_data,
    artifact: ModelArtifacts,
) -> dict:
    logger.info("Starting customer churn prediction.")

    try:
        features = prepare_features(customer_data)
        processed_features = artifact.preprocessor.transform(features)

        probability = artifact.model.predict_proba(
            processed_features
        )[:, 1]

        # prediction = (probability >= artifact.threshold).astype(int)
        prediction = apply_threshold(
            probability,
            artifact.threshold,
        )

        result = {
            "churn_probability": float(probability[0]),
            "prediction": int(prediction[0]),
        }

        logger.info("Customer churn prediction completed successfully.")

        return result

    except Exception:
        logger.exception("Customer churn prediction failed.")
        raise