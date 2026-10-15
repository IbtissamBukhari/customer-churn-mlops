import logging

import logging_config
import mlflow
import pandas as pd

from src.config import (
    MLFLOW_DB_PATH,
    RAW_DATA_PATH,
    ARTIFACT_PATH,
)
from src.models.mlflow_pyfunc import ChurnPredictionModel


logger = logging.getLogger(__name__)


def register_model():
    logger.info("Starting churn model registration.")

    try:
        # Step 1: Connect to our MLflow tracking database.
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

        # Step 2: Select our existing experiment.
        mlflow.set_experiment("customer-churn-mlops-v2")

        # Step 3: Load the dataset.
        df = pd.read_csv(RAW_DATA_PATH)

        logger.info("Dataset loaded successfully for model registration.")

        # Step 4: Prepare one example customer without the target column.
        # input_example = df.drop(columns=["Churn"]).iloc[[0]]
        input_example = df.drop(columns=["Churn"]).iloc[[0]].copy()

        input_example["TotalCharges"] = pd.to_numeric(
            input_example["TotalCharges"],
            errors="coerce",
        )

        logger.info(
            "Input example TotalCharges dtype: %s",
            input_example["TotalCharges"].dtype,
        )

        # Step 5: Start an MLflow run and log the model.
        with mlflow.start_run(run_name="register_churn_model"):
            model_info = mlflow.pyfunc.log_model(
                name="churn-predictor",
                python_model=ChurnPredictionModel(),
                artifacts={
                    "churn_artifact": str(ARTIFACT_PATH),
                },
                input_example=input_example,
                registered_model_name="CustomerChurnModel",
            )

            logger.info("Model registered successfully.")
            logger.info("Model URI: %s", model_info.model_uri)

        logger.info("Churn model registration completed successfully.")

    except Exception:
        logger.exception("Churn model registration failed.")
        raise


if __name__ == "__main__":
    register_model()