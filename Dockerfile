FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first so the layer is cached across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code, model artifact, and config.
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
