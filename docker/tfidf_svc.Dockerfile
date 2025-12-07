# docker/tfidf_svc.Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 1. Créer le dossier pour garder la structure propre
RUN mkdir -p src/services/tfidf_service

# 2. Copier les fichiers nécessaires
COPY src/services/tfidf_service/app.py src/services/tfidf_service/
# Install curl with minimal memory usage + clean immediately
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements/base.txt requirements/tfidf.txt ./

# 3. Installer tout d’un coup (requirements + nltk + stopwords)
#    → une seule couche = image plus petite + plus rapide
RUN pip install --no-cache-dir -r base.txt -r tfidf.txt \
    && pip install --no-cache-dir nltk \
    && python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data')"

# 4. Variables d’environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/usr/share/nltk_data

EXPOSE 8000

# Healthcheck propre (wget n’existe pas par défaut dans python-slim → on utilise curl ou python)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import requests; exit(0) if requests.get('http://localhost:8000/health', timeout=2).status_code == 200 else exit(1)"

CMD ["uvicorn", "src.services.tfidf_service.app:app", "--host", "0.0.0.0", "--port", "8000"]



