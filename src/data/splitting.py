from typing import Tuple
import logging
import logging_config
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (RANDOM_STATE,TEST_SIZE,VALIDATION_SIZE,TRAIN_SIZE)
from src.exceptions import DataSplittingError,DataValidationError
logger = logging.getLogger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split features and target into training, validation, and test sets."""

    logger.info("Starting data splitting.")

    # Step 1: Validate input types.
    if not isinstance(X, pd.DataFrame):
        logger.error("Data splitting failed: X must be a pandas DataFrame.")
        raise DataValidationError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        logger.error("Data splitting failed: y must be a pandas Series.")
        raise DataValidationError("y must be a pandas Series.")

    # Step 2: Validate that inputs contain data.
    if X.empty or y.empty:
        logger.error("Data splitting failed: X or y is empty.")
        raise DataValidationError("X and y must not be empty.")

    # Step 3: Ensure every feature row has a target label.
    if len(X) != len(y):
        logger.error(
            "Data splitting failed: X has %d rows, but y has %d rows.",
            len(X),
            len(y),
        )
        raise DataValidationError(
            "X and y must contain the same number of rows."
        )

    # Step 4: Validate target values.
    if y.isna().any():
        logger.error("Data splitting failed: target contains missing values.")
        raise DataValidationError("Target contains missing values.")

    # Step 5: Validate split configuration.
    split_sizes = (TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE)

    if any(size <= 0 or size >= 1 for size in split_sizes):
        logger.error("Data splitting failed: split sizes must be between 0 and 1.")
        raise DataValidationError(
            "TRAIN_SIZE, VALIDATION_SIZE, and TEST_SIZE "
            "must each be between 0 and 1."
        )

    if not abs(sum(split_sizes) - 1.0) < 1e-9:
        logger.error("Data splitting failed: split sizes must sum to 1.")
        raise DataValidationError("Split sizes must sum to 1.")

    # Stratification requires at least two target classes.
    if y.nunique() < 2:
        logger.error(
            "Data splitting failed: target must contain at least two classes."
        )
        raise DataValidationError(
            "Target must contain at least two classes for stratified splitting."
        )

    try:
        # Step 6: Create the training set and temporary set.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE + TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        # Step 7: Divide the temporary set into validation and test sets.
        test_fraction = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_fraction,
            stratify=y_temp,
            random_state=RANDOM_STATE,
        )

    except ValueError as exc:
        logger.exception("Data splitting failed because the input data is unsuitable.")
        raise DataSplittingError(
            f"Unable to split the dataset: {exc}"
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error occurred during data splitting.")
        raise DataSplittingError(
            "An unexpected error occurred while splitting the dataset."
        ) from exc

    logger.info("Data splitting completed: train=%d, validation=%d, test=%d.",
        len(X_train),len(X_val), len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test
