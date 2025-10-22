#for building run: docker build -t tfidf_svc -f docker/tfidf_svc.Dockerfile .

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

COPY src/services/tfidf_service ./src/services/tfidf_service


ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=tfidf_svm_model
ENV MODEL_STAGE=Latest
ENV TRANSFORMERS_SERVICE_URL=http://transformer:8001/scrub_pii
ENV PYTHONUNBUFFERED=1
ENV PORT=8000


EXPOSE 8000
CMD ["uvicorn", "src.services.tfidf_svc.app:app", "--host", "0.0.0.0", "--port", "8000"]
