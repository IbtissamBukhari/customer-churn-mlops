
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def configure_logging():
    """Configure shared logging for the whole project."""

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove only handlers created by this configuration.
    for handler in root_logger.handlers[:]:
        if getattr(handler, "_churn_logging_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler._churn_logging_handler = True

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler._churn_logging_handler = True

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


configure_logging()
