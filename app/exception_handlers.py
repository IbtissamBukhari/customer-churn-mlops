
from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions import ModelPredictionError
from fastapi.exceptions import RequestValidationError


async def model_prediction_exception_handler(
    request: Request,
    exc: ModelPredictionError,) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction failed due to an internal server error."
        },
    )

async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request data. Please check your input."
        },
    )
