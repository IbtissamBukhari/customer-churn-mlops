````markdown
# Customer Churn Prediction — Data Science & MLOps

![CI/CD](https://github.com/IbtissamBukhari/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)

An end-to-end, production-oriented machine learning project that predicts customer churn and serves predictions through a deployed REST API. The project combines data science, model evaluation, experiment tracking, API development, automated testing, containerization, and CI/CD.

<p align="center">
  <a href="https://customer-churn-mlops-e3k1.onrender.com/docs"><strong>Live API Documentation</strong></a>
  ·
  <a href="https://customer-churn-mlops-e3k1.onrender.com/health"><strong>API Health</strong></a>
  ·
  <a href="https://hub.docker.com/r/ibtissambukhari/customer-churn-mlops"><strong>Docker Hub</strong></a>
</p>

> **Note:** The API is hosted on Render's free tier. The first request after a period of inactivity may take up to a minute while the service wakes up.

**Tech stack:** Python · pandas · scikit-learn · MLflow · FastAPI · pytest · Docker · GitHub Actions · Render

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Machine Learning Approach](#machine-learning-approach)
- [Live API](#live-api)
- [Example Prediction](#example-prediction)
- [Run Locally](#run-locally)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Project Structure](#project-structure)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Author](#author)

---

## Overview

Customer churn occurs when a customer stops using a company's products or services. Predicting churn helps businesses identify customers who may leave and prioritize retention efforts.

This project uses the IBM Telco Customer Churn dataset to build a machine learning system that estimates the likelihood of a customer leaving.

The trained model is served through FastAPI, packaged into a Docker image, and deployed on Render. GitHub Actions automates testing, Docker image publishing, and deployment.

The system returns:

- Churn probability
- Binary churn prediction
- A decision based on a selected probability threshold

---

## Key Features

- Data ingestion and validation
- Exploratory data analysis (EDA)
- Data preprocessing and feature preparation
- Model training and evaluation
- Probability threshold analysis
- MLflow experiment tracking and model registration
- Reusable model artifacts and prediction logic
- FastAPI REST API
- Health and readiness endpoints
- Application logging and error handling
- Automated tests using pytest
- Docker-based deployment
- GitHub Actions CI/CD automation
- Live deployment on Render

---

## System Architecture

```text
Dataset → Ingestion → Validation → EDA → Preprocessing → Training & Evaluation
        → Threshold Selection → MLflow Tracking → Model Artifacts
        → FastAPI → Docker Image → Render → Prediction API
```

---

## Dataset

**IBM Telco Customer Churn**

The dataset contains customer demographics, subscribed services, account information, billing details, and a churn target.

The original dataset contains **7,043 customer records** and **21 columns**.

Examples of available features include:

- `tenure` — how long the customer has stayed with the company
- `Contract` — contract type
- `MonthlyCharges` — monthly customer charges
- `TotalCharges` — total customer charges
- `InternetService` — type of internet service
- `TechSupport` — whether technical support is included
- `PaymentMethod` — customer payment method
- `Churn` — whether the customer left

The target column is `Churn`.

The raw dataset is not committed to Git. Place the downloaded CSV at:

```text
data/raw/Telco-Customer-Churn.csv
```

Dataset source: IBM Telco Customer Churn sample data.

---

## Machine Learning Approach

The project follows a structured machine learning workflow:

1. **Data understanding:** Explore customer characteristics, missing values, and churn patterns.
2. **Data validation:** Check required columns, target values, and data quality.
3. **Preprocessing:** Prepare numerical and categorical features for model training.
4. **Model training:** Train and evaluate candidate models.
5. **Threshold selection:** Evaluate how different probability thresholds affect churn predictions.
6. **Experiment tracking:** Use MLflow to track experiments and manage registered model versions.
7. **Artifact-based inference:** Load exported model artifacts for API predictions.

<!--
### Model Results

| Model | ROC-AUC | Recall (churn) | Precision (churn) | F1 (churn) |
|---|---|---|---|---|
| Model A | 0.xx | 0.xx | 0.xx | 0.xx |
| Model B | 0.xx | 0.xx | 0.xx | 0.xx |

**Selected model:** ...
-->

### Decision Threshold

The deployed prediction threshold is **0.30**.

The API uses this threshold to convert a predicted probability into a binary result:

- Probability **≥ 0.30** → predicted churn (`1`)
- Probability **< 0.30** → predicted no churn (`0`)

The threshold represents a decision choice, not a guarantee that a customer will leave.

---

## Live API

**Base URL:** https://customer-churn-mlops-e3k1.onrender.com

**Interactive Swagger documentation:** https://customer-churn-mlops-e3k1.onrender.com/docs

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks whether the application is alive |
| `/ready` | GET | Checks whether the model is available |
| `/predict` | POST | Returns a customer churn prediction |

### Health Check

```http
GET /health
```

```json
{
  "status": "alive"
}
```

### Readiness Check

```http
GET /ready
```

```json
{
  "status": "ready",
  "model": "available"
}
```

---

## Example Prediction

Send a `POST` request to `/predict` with a customer profile.

**Example request:**

```json
{
  "customerID": "TEST-001",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 171.0
}
```

**Example response:**

```json
{
  "churn_probability": 0.7250805497169495,
  "prediction": 1
}
```

In this example, the model estimates a churn probability of approximately **72.51%**. Because the probability exceeds the selected threshold of 0.30, the system returns `prediction: 1`.

This is a model estimate, not a guarantee of future customer behavior.

Try your own request using the [interactive Swagger API documentation](https://customer-churn-mlops-e3k1.onrender.com/docs).

---

## Run Locally

### Prerequisites

- Python 3.11
- Git
- Conda or another Python virtual environment tool

### 1. Clone the Repository

```bash
git clone https://github.com/IbtissamBukhari/customer-churn-mlops.git
cd customer-churn-mlops
```

### 2. Create and Activate an Environment

```bash
conda create -n churn-mlops python=3.11
conda activate churn-mlops
```

### 3. Install Dependencies

To run the API only:

```bash
python -m pip install -r requirements.txt
```

To run tests and development tools:

```bash
python -m pip install -r requirements-dev.txt
```

### 4. Start the API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Open the API Documentation

```text
http://127.0.0.1:8000/docs
```

> **Note:** The application expects the required model artifacts to be available in the locations configured by the project. The raw training dataset is not required to serve predictions when the exported inference artifacts are present.

---

## Testing

The project uses **pytest** for automated testing.

Run the test suite from the repository root:

```bash
python -m pytest -v
```

Tests cover data validation, preprocessing, model-related logic, and API behavior.

The GitHub Actions workflow runs the test suite automatically on pushes and pull requests targeting `master`.

---

## Docker

The project is packaged as a Docker image so the API and its model artifacts run in a consistent environment.

**Docker Hub image:**

```text
ibtissambukhari/customer-churn-mlops:latest
```

**Image repository:** https://hub.docker.com/r/ibtissambukhari/customer-churn-mlops

The image includes the application code, runtime dependencies, and exported model artifacts. It loads the trained artifacts for inference; it does not retrain the model when the container starts.

### Pull and Run the Published Image

```bash
docker pull ibtissambukhari/customer-churn-mlops:latest
docker run --rm -p 8000:8000 ibtissambukhari/customer-churn-mlops:latest
```

### Build and Run Locally

```bash
docker build -t customer-churn-mlops .
docker run --rm -p 8000:8000 customer-churn-mlops
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## CI/CD Pipeline

GitHub Actions automates testing, image publishing, and deployment.

```text
Code push to master
        │
        ▼
  GitHub Actions
        │
        ▼
Install dependencies
        │
        ▼
  Run pytest tests
        │
   Tests pass?
    ┌───┴───┐
   No      Yes
    │       │
  Stop   Build Docker image
            │
            ▼
     Push to Docker Hub
            │
            ▼
   Trigger Render deploy
            │
            ▼
     Live API updated
```

### On Pushes to `master`

1. Check out the repository.
2. Set up Python and install development dependencies.
3. Run automated tests.
4. Build the Docker image.
5. Publish the image to Docker Hub.
6. Trigger deployment on Render.

### On Pull Requests Targeting `master`

The workflow runs automated tests only. The publishing and deployment job is restricted to pushes to `master`.

### Required GitHub Actions Secrets

| Secret | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `RENDER_DEPLOY_HOOK_URL` | Render deploy hook used to trigger deployment |

These values are configured in the repository's GitHub Actions secrets and must never be committed to source control.

---

## Project Structure

```text
customer-churn-mlops/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── tests/
├── deployment/
│   └── artifacts/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── logging_config.py
├── .gitignore
└── README.md
```

The exact contents of individual directories may evolve as the project develops.

---

## Limitations and Future Improvements

- The model provides probabilities based on historical data; it cannot guarantee future customer behavior.
- The selected threshold should be assessed against business costs, such as the cost of missing a customer likely to churn versus the cost of unnecessary retention offers.
- The Render free instance can spin down after inactivity, causing cold-start delays.
- Monitoring for data drift, model performance degradation, and changes in customer behavior can be expanded.
- Automated model retraining and a more comprehensive model evaluation pipeline are potential future improvements.
- Deployment can be hardened further by deploying immutable Docker image digests rather than relying only on the mutable `latest` tag.

---

## Author

**Ibtissam Bukhari**

GitHub: [IbtissamBukhari](https://github.com/IbtissamBukhari)

This project demonstrates an end-to-end machine learning workflow, from data exploration and model development to API serving, containerization, automated testing, and deployment.
````