FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow>=2.0.0,<4.0.0"

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", \
    "--backend-store-uri", "sqlite:////data/mlflow.db", \
    "--default-artifact-root", "/data/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000"]
