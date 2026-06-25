FROM python:3.12-slim

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY api_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_service/main.py .

USER appuser

ENV APP_PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-8000}"]
