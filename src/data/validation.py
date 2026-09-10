from typing import List

import pandas as pd


class DataValidationError(Exception):
    """Raised when the input data fails validation."""


EXPECTED_COLUMNS: List[str] = [
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


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that the dataset contains the expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Raises
    ------
    DataValidationError
        If required columns are missing or unexpected columns are present.
    """

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
        raise DataValidationError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        raise DataValidationError(
            f"Unexpected columns found: {unexpected_columns}"
        )


def validate_target_column(
    df: pd.DataFrame,
    target_column: str = "Churn",
) -> None:
    """
    Validate the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    target_column : str
        Name of the target column.

    Raises
    ------
    DataValidationError
        If the target column is missing or contains unexpected values.
    """

    if target_column not in df.columns:
        raise DataValidationError(
            f"Target column '{target_column}' is missing."
        )

    allowed_values = {"Yes", "No"}

    actual_values = set(
        df[target_column].dropna().unique()
    )

    unexpected_values = actual_values - allowed_values

    if unexpected_values:
        raise DataValidationError(
            f"Unexpected values found in target column "
            f"'{target_column}': {unexpected_values}"
        )


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    """
    Validate ranges of important numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Raises
    ------
    DataValidationError
        If important numeric columns contain invalid values.
    """

    if "tenure" not in df.columns:
        raise DataValidationError(
            "Column 'tenure' is missing."
        )

    if "MonthlyCharges" not in df.columns:
        raise DataValidationError(
            "Column 'MonthlyCharges' is missing."
        )

    if (df["tenure"] < 0).any():
        raise DataValidationError(
            "Invalid values found in 'tenure'. "
            "Tenure cannot be negative."
        )

    if (df["MonthlyCharges"] < 0).any():
        raise DataValidationError(
            "Invalid values found in 'MonthlyCharges'. "
            "Monthly charges cannot be negative."
        )


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Run all raw-data validation checks.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Raises
    ------
    DataValidationError
        If any validation check fails.
    """

    validate_columns(df)
    validate_target_column(df)
    validate_numeric_ranges(df)