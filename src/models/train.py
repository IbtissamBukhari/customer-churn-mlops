import logging

import logging_config
import numpy as np
from xgboost import XGBClassifier

from src.config import XGB_MODEL_PARAMS


logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    logger.info("Starting XGBoost model training.")

    try:
        model = XGBClassifier(
            **XGB_MODEL_PARAMS
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.info("XGBoost model training completed successfully.")

        return model

    except Exception:
        logger.exception("XGBoost model training failed.")
        raise