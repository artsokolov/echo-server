FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch first (separate step to use PyTorch index)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    "fastapi[standard]>=0.115.0" \
    "uvicorn[standard]>=0.20.0" \
    "transformers>=4.40.0"

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
