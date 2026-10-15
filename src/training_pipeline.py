import logging
from pathlib import Path

import mlflow
import logging_config

from src.data.ingestion import load_raw_data
from src.data.validation import validate_raw_data
from src.data.splitting import split_data
from src.features.preprocessing import (
    prepare_features,
    prepare_target,
    create_preprocessor,
)
from src.models.train import train_model
from src.models.evaluate import (
    predict_probabilities,
    apply_threshold,
    calculate_metrics,
)
from src.models.artifacts import ModelArtifacts, save_artifacts
from src.config import (
    DECISION_THRESHOLD,
    MODEL_TYPE,
    MODEL_VERSION,
    TARGET_COLUMN,
    MLFLOW_DB_PATH,
    XGB_MODEL_PARAMS,
)


logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("customer-churn-mlops-v2")


def run_training_pipeline():
    """Run the complete customer churn training pipeline."""

    logger.info("Starting the customer churn training pipeline.")

    try:
        with mlflow.start_run():

            # Step 1: Log training configuration.
            mlflow.log_params(XGB_MODEL_PARAMS)
            mlflow.log_param("decision_threshold", DECISION_THRESHOLD)

            # Step 2: Load and validate the raw data.
            df = load_raw_data()
            validate_raw_data(df)

            logger.info("Raw data loaded and validated successfully.")

            # Step 3: Prepare features and target.
            X = prepare_features(df)
            y = prepare_target(df[TARGET_COLUMN])

            # Step 4: Split the data.
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = split_data(X, y)

            logger.info("Data splitting completed successfully.")

            # Step 5: Fit the preprocessor using training data only.
            preprocessor = create_preprocessor()
            preprocessor.fit(X_train)

            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)

            logger.info("Feature preprocessing completed successfully.")

            # Step 6: Train the model.
            model = train_model(
                X_train_processed,
                y_train.to_numpy(),
            )

            logger.info("Model training completed successfully.")

            # Step 7: Evaluate on validation data.
            val_probabilities = predict_probabilities(
                model, X_val_processed
            )
            val_predictions = apply_threshold(
                val_probabilities, DECISION_THRESHOLD
            )

            validation_metrics = calculate_metrics(
                y_val.to_numpy(),
                val_predictions,
                val_probabilities,
            )

            mlflow.log_metrics({
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            })

            logger.info("Validation evaluation completed.")

            # Step 8: Evaluate on test data.
            test_probabilities = predict_probabilities(
                model, X_test_processed
            )
            test_predictions = apply_threshold(
                test_probabilities, DECISION_THRESHOLD
            )

            test_metrics = calculate_metrics(
                y_test.to_numpy(),
                test_predictions,
                test_probabilities,
            )

            mlflow.log_metrics({
                f"test_{key}": value
                for key, value in test_metrics.items()
            })

            logger.info("Test evaluation completed.")

            # Step 9: Create and save model artifacts.
            artifact = ModelArtifacts(
                model=model,
                preprocessor=preprocessor,
                threshold=DECISION_THRESHOLD,
                metadata={
                    "model_type": MODEL_TYPE,
                    "model_version": MODEL_VERSION,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                },
            )

            project_root = Path(__file__).resolve().parents[1]
            artifact_path = (
                project_root / "models" / "churn_model.joblib"
            )

            save_artifacts(artifact, artifact_path)
            mlflow.log_artifact(str(artifact_path))

            logger.info(
                "Model artifacts saved successfully at %s.",
                artifact_path,
            )

            logger.info("Training pipeline completed successfully.")

            return {
                "artifact_path": artifact_path,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            }

    except Exception:
        logger.exception("Customer churn training pipeline failed.")
        raise


if __name__ == "__main__":
    run_training_pipeline()