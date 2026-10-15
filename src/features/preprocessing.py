import pandas as pd
from src.config import TARGET_COLUMN,ID_COLUMN,NUMERIC_FEATURES,CATEGORICAL_FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
import logging_config
logger = logging.getLogger(__name__)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and handle expected blanks."""

    logger.info("Starting TotalCharges cleaning: rows=%d", len(df))

    df = df.copy()

    if "TotalCharges" not in df.columns:
        logger.error("Required feature 'TotalCharges' is missing.")
        raise ValueError("Required feature 'TotalCharges' is missing.")

    if "tenure" not in df.columns:
        logger.error("Required feature 'tenure' is missing.")
        raise ValueError("Required feature 'tenure' is missing.")

    total_charges = pd.to_numeric(  df["TotalCharges"].astype(str).str.strip().replace("", pd.NA),errors="coerce")

    expected_blank = total_charges.isna() & (df["tenure"] == 0)
    unexpected_invalid = total_charges.isna() & ~expected_blank

    if unexpected_invalid.any():
        logger.error("TotalCharges cleaning failed: invalid or missing values "
            "found for customers whose tenure is not zero; count=%d",int(unexpected_invalid.sum()) )
        raise ValueError("TotalCharges contains invalid or missing values ""for customers whose tenure is not zero.")

    expected_blank_count = int(expected_blank.sum())

    df["TotalCharges"] = total_charges.fillna(0)

    logger.info("TotalCharges cleaning completed: expected blanks replaced=%d",expected_blank_count)

    return df



def prepare_target(y: pd.Series) -> pd.Series:
    """Convert churn labels into binary values."""

    logger.info("Starting target preparation: rows=%d", len(y))

    allowed_values = {"No", "Yes"}

    if y.isna().any():
        logger.error("Target preparation failed: target contains missing values.")
        raise ValueError("Target contains missing values.")

    unexpected_values = set(y.unique()) - allowed_values

    if unexpected_values:
        logger.error("Target preparation failed: unexpected target values=%s",unexpected_values)
        raise ValueError(f"Unexpected target values: {unexpected_values}" )

    prepared_target = y.map({"No": 0, "Yes": 1})

    logger.info("Target preparation completed successfully.")

    return prepared_target



def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare input data for model training or prediction."""

    logger.info("Starting feature preparation: rows=%d, columns=%d",df.shape[0],df.shape[1])

    if df.empty:
        logger.error("Feature preparation failed: input DataFrame is empty.")
        raise ValueError("Input DataFrame cannot be empty.")

    if ID_COLUMN not in df.columns:
        logger.error("Feature preparation failed: identifier column '%s' is missing.",ID_COLUMN)
        raise ValueError(f"Required identifier column '{ID_COLUMN}' is missing.")

    if "TotalCharges" not in df.columns:
        logger.error("Feature preparation failed: required feature 'TotalCharges' is missing.")
        raise ValueError("Required feature 'TotalCharges' is missing.")

    df = clean_total_charges(df)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
        logger.info("Dropped target column '%s' from features.", TARGET_COLUMN)

    df = df.drop(columns=[ID_COLUMN])

    logger.info("Feature preparation completed: rows=%d, columns=%d",df.shape[0],df.shape[1])

    return df



def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer for model features."""

    logger.info("Creating feature preprocessor: numeric_features=%d,"
        "categorical_features=%d",len(NUMERIC_FEATURES),len(CATEGORICAL_FEATURES))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    logger.info("Feature preprocessor created successfully.")

    return preprocessor
