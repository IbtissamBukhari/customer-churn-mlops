class DataIngestionError(Exception):
    """Raised when the raw dataset cannot be loaded."""

class DataValidationError(Exception):
    """Raised when the input data fails validation."""

class DataSplittingError(Exception):
    """Raised when the dataset cannot be split successfully."""