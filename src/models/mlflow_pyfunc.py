import logging

import logging_config
import joblib
import pandas as pd
import mlflow.pyfunc

from src.models.predict import predict_customer


logger = logging.getLogger(__name__)


class ChurnPredictionModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        logger.info("Loading churn model artifacts.")

        try:
            self.artifact = joblib.load(
                context.artifacts["churn_artifact"]
            )

            logger.info("Churn model artifacts loaded successfully.")

        except Exception:
            logger.exception("Failed to load churn model artifacts.")
            raise

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: dict = None,
    ) -> pd.DataFrame:
        logger.info(
            "Starting churn predictions for %d customers.",
            len(model_input),
        )

        try:
            results = []

            for _, customer in model_input.iterrows():
                customer_data = customer.to_frame().T

                result = predict_customer(
                    customer_data,
                    self.artifact,
                )

                results.append(result)

            predictions = pd.DataFrame(results)

            logger.info(
                "Churn predictions completed for %d customers.",
                len(predictions),
            )

            return predictions

        except Exception:
            logger.exception("Churn prediction failed.")
            raise