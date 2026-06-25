FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi[standard]>=0.115.0" \
    "uvicorn[standard]>=0.20.0" \
    "httpx>=0.27.0"

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
