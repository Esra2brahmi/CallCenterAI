# docker/transformersService.Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy service files
RUN mkdir -p src/services/transformer_service
COPY src/services/transformer_service/serviceFromMLFlow.py src/services/transformer_service/

# Copy requirements
COPY requirements/base.txt requirements/transformer.txt ./

# Install PyTorch CPU, Transformers, NLTK, MLflow
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        transformers==4.41.0 \
        mlflow==3.4.0 \
        nltk && \
    python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data')" && \
    pip install --no-cache-dir -r base.txt -r transformer.txt

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/usr/share/nltk_data

# Expose port
EXPOSE 8000

# Healthcheck with longer startup period
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import requests; exit(0) if requests.get('http://localhost:8000/health', timeout=5).status_code == 200 else exit(1)"

# Run FastAPI
CMD ["uvicorn", "src.services.transformer_service.serviceFromMLFlow:app", "--host", "0.0.0.0", "--port", "8000"]
