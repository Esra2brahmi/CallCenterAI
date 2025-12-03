FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p src/services/transformer_service

COPY src/services/transformer_service/serviceFromMLFlow.py ./src/services/transformer_service/

COPY src/services/transformer_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nltk \
    && python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data')"


RUN python -m nltk.downloader stopwords



EXPOSE 8000

CMD ["uvicorn", "src.services.transformer_service.serviceFromMLFlow:app", "--host", "0.0.0.0", "--port", "8000"]
