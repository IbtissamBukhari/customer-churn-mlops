FROM python:3.14-slim
WORKDIR /workspaces/customer-churn-mlops

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY logging_config.py .

COPY mlflow.db mlflow.db
COPY mlruns/ mlruns/
COPY models/ models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
