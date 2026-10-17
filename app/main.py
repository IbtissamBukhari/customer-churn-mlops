import mlflow
import math
import pandas as pd
from pathlib import Path
from typing import Literal
from src.config import PROJECT_ROOT
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field,model_validator
from fastapi.exceptions import RequestValidationError
import logging
import logging_config
from app.exceptions import ModelPredictionError
from app.exception_handlers import (model_prediction_exception_handler,request_validation_exception_handler)

logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Churn Prediction API")
app.add_exception_handler( ModelPredictionError,model_prediction_exception_handler)
app.add_exception_handler(RequestValidationError,request_validation_exception_handler)

model_path = PROJECT_ROOT / "deployment"

model = None
try:
    model = mlflow.pyfunc.load_model("deployment")
    logger.info("MLflow model loaded successfully.")

except Exception as exc:
    model = None
    logger.exception(
        "Failed to load the MLflow model. "
        "The API will start but will not be ready for predictions.")

class Customer(BaseModel):
    customerID: str

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]

    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal["Bank transfer (automatic)","Credit card (automatic)",
                           "Electronic check","Mailed check"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_phone_service(self):
        if self.PhoneService == "No":
            if self.MultipleLines != "No phone service":
                raise ValueError(
                    "When PhoneService is 'No', "
                    "MultipleLines must be 'No phone service'."
                )

        elif self.MultipleLines == "No phone service":
            raise ValueError(
                "When PhoneService is 'Yes', "
                "MultipleLines cannot be 'No phone service'." )

        return self



@app.get("/health")
def health_check():
    return {"status": "alive"}

@app.get("/ready")
def readiness_check():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        )

    return {"status": "ready", "model": "available"}

@app.post("/predict")
def predict_customer(customer: Customer):
    logger.info("Prediction request received.")
    if model is None:
            logger.error(
                "Prediction rejected because the MLflow model is unavailable."
            )
            raise HTTPException(
                status_code=503,
                detail="Prediction service is temporarily unavailable.",
            )
    try:
        # Convert the customer data to a DataFrame
        customer_df = pd.DataFrame([customer.model_dump()])
        # Make a prediction using the loaded model
        result = model.predict(customer_df)
        logger.info("Prediction completed successfully.")
        prediction_response = result.to_dict(orient="records")[0]
        churn_probability = float( prediction_response["churn_probability"])

        if not math.isfinite(churn_probability) or not 0 <= churn_probability <= 1:
            raise ModelPredictionError( "Model returned an invalid churn probability.")

        prediction_response["churn_probability"] = churn_probability
        logger.info(
           "Prediction completed: prediction=%s churn_probability=%.4f",
            prediction_response["prediction"],
            prediction_response["churn_probability"]
            )
        return prediction_response
    except ModelPredictionError:
        logger.exception("Model prediction validation failed.")
        raise

    except Exception as exc:
        logger.exception("Unexpected prediction error: %s", exc)
        raise ModelPredictionError(
            "Prediction failed due to an internal server error."
        ) from exc
