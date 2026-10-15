
import logging
from src.exceptions import DataValidationError
import logging_config
import pandas as pd
import math

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: list[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def _raise_validation_error(message: str) -> None:
    """Log a validation failure and raise a consistent exception."""
    logger.error("Data validation failed: %s", message)
    raise DataValidationError(message)


def validate_columns(df: pd.DataFrame) -> None:
    """Validate the expected dataset columns."""

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        _raise_validation_error(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        _raise_validation_error(
            f"Unexpected columns found: {unexpected_columns}"
        )


def validate_target_column(
    df: pd.DataFrame,
    target_column: str = "Churn",
) -> None:
    """Validate the target column and its values."""

    if target_column not in df.columns:
        _raise_validation_error(
            f"Target column '{target_column}' is missing."
        )

    if df[target_column].isna().any():
        _raise_validation_error(
            f"Target column '{target_column}' contains missing values."
        )

    allowed_values = {"Yes", "No"}

    actual_values = set(df[target_column].unique())
    unexpected_values = actual_values - allowed_values

    if unexpected_values:
        _raise_validation_error(
            f"Unexpected values found in target column "
            f"'{target_column}': {unexpected_values}"
        )


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    """Validate numeric types and ranges for important columns."""

    numeric_ranges = {
        "tenure": (0, 72),
        "MonthlyCharges": (0, float("inf")),
    }

    for column, (minimum, maximum) in numeric_ranges.items():
        if column not in df.columns:
            _raise_validation_error(
                f"Required numeric column '{column}' is missing."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            _raise_validation_error(
                f"Column '{column}' must contain numeric data."
            )

        if df[column].isna().any():
            _raise_validation_error(
                f"Column '{column}' contains missing values."
            )

        if not df[column].between(minimum, maximum).all():
            _raise_validation_error(
                f"Column '{column}' contains values outside "
                f"the allowed range [{minimum}, {maximum}]."
            )
        if not df[column].map(math.isfinite).all():
           _raise_validation_error(f"Column '{column}' contains non-finite values.")

        if not df[column].map( lambda value: abs(value) != float("inf") ).all():
            _raise_validation_error(f"Column '{column}' contains infinite values." )


def validate_raw_data(df: pd.DataFrame) -> None:
    """Run all raw-data validation checks."""

    logger.info(
        "Starting raw-data validation: rows=%d, columns=%d",
        df.shape[0],
        df.shape[1],
    )

    validate_columns(df)
    validate_target_column(df)
    validate_numeric_ranges(df)

    logger.info("Raw-data validation completed successfully.")
