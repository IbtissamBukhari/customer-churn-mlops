# Customer Churn Prediction — Data Science + MLOps

An end-to-end, production-oriented customer churn prediction system built to practice Data Science, Machine Learning, and MLOps.

## Project Goal

Build a machine learning system that predicts whether a customer is likely to churn.

The final system will provide:

* Churn probability
* Churn prediction
* Decision threshold
* Business-oriented interpretation

## Planned ML Workflow

```text
Raw Data
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Exploratory Data Analysis
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Threshold Selection
   ↓
Final Model
   ↓
MLflow
   ↓
FastAPI
   ↓
Testing
   ↓
Docker
   ↓
CI/CD
   ↓
Deployment
   ↓
Monitoring
   ↓
Retraining
```

## Project Structure

```text
customer-churn-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   └── models/
│
├── app/
├── tests/
├── models/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Development Environment

* Python 3.11
* Conda
* VS Code
* Git

## Current Status

Project foundation and Python environment setup completed.

Next stage:

**Data source selection and data ingestion planning.**
# Data Directory

This directory contains the datasets used by the Customer Churn Prediction project.

## Raw Data

Raw data is stored in:

```text
data/raw/
```

Raw data should be treated as immutable and should not be manually modified.

## Processed Data

Generated or transformed datasets may be stored in:

```text
data/processed/
```

Processed data should be reproducible from the raw data and the project code whenever practical.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

Source:

IBM Telco Customer Churn / IBM Cognos Analytics sample data.

The dataset contains customer information related to demographics, services, account information, and churn.

The raw dataset is not committed to Git.
