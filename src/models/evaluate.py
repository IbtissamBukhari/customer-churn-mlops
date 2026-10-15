import logging

import logging_config
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


def predict_probabilities(
    model,
    X: np.ndarray,
) -> np.ndarray:
    logger.info("Generating prediction probabilities.")

    try:
        probabilities = model.predict_proba(X)[:, 1]
        logger.info("Prediction probabilities generated successfully.")
        return probabilities

    except Exception:
        logger.exception("Failed to generate prediction probabilities.")
        raise


def apply_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    logger.info("Applying decision threshold: %s.", threshold)

    try:
        predictions = (probabilities >= threshold).astype(int)
        logger.info("Decision threshold applied successfully.")
        return predictions

    except Exception:
        logger.exception("Failed to apply decision threshold.")
        raise


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    logger.info("Calculating evaluation metrics.")

    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probabilities),
        }

        logger.info("Evaluation metrics calculated successfully.")
        return metrics

    except Exception:
        logger.exception("Failed to calculate evaluation metrics.")
        raise


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    logger.info("Starting threshold evaluation.")

    try:
        results = []

        for threshold in thresholds:
            predictions = apply_threshold(
                probabilities,
                threshold,
            )

            metrics = calculate_metrics(
                y_true,
                predictions,
                probabilities,
            )

            metrics["threshold"] = threshold
            results.append(metrics)

        logger.info(
            "Threshold evaluation completed for %d thresholds.",
            len(thresholds),
        )

        return results

    except Exception:
        logger.exception("Threshold evaluation failed.")
        raise