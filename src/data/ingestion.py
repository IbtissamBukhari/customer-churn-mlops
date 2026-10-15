from pathlib import Path
import logging
import logging_config
import pandas as pd
from src.config import RAW_DATA_PATH
from src.exceptions import DataIngestionError
logger = logging.getLogger(__name__)

def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw customer churn dataset.

    Parameters
    ----------
    file_path : Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded customer churn dataset.

    Raises
    ------
    DataIngestionError
        If the dataset is missing or cannot be read as a CSV.
    """
    logger.info("Loading raw dataset from: %s", file_path)
    if not file_path.exists():
        logger.error("Raw dataset file not found: %s", file_path)
        raise DataIngestionError(
        f"Raw dataset file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.exception("Failed to read CSV file: %s", file_path)

        raise DataIngestionError(
          f"Failed to read CSV file: {file_path}"
          ) from exc

    logger.info("Raw dataset loaded successfully: rows=%d, columns=%d",df.shape[0],df.shape[1])
    return df
