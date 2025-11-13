FROM python:3.12-slim


WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt


COPY src/services/transformers_service ./src/services/transformers_service

EXPOSE 8000

CMD ["uvicorn", "src.services.transformers_service.app:app", "--host" ]

