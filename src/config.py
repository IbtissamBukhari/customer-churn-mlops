from pathlib import Path
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ( PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv" )
ARTIFACT_PATH = (PROJECT_ROOT / "models"/ "churn_model.joblib")
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

# Data splitting
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Training
RANDOM_STATE = 42

# Model settings
DECISION_THRESHOLD = 0.30
MODEL_TYPE = "XGBoost"
MODEL_VERSION = "1.0"
XGB_MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}

# Target column
TARGET_COLUMN = "Churn"

# Feature columns
ID_COLUMN = "customerID"


# Target and identifier columns
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Feature columns
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]
