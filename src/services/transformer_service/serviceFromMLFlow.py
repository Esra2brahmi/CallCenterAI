# src/services/transformer_service/serviceFromMLFlow.py
import logging
import os
import re
import string
from typing import Dict

import requests
from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from nltk.corpus import stopwords
from pydantic import BaseModel
import torch
import uvicorn

# -----------------------------
# CONFIG
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "CallCenterTransformer")
TRANSFORMERS_SERVICE_URL = os.getenv("TRANSFORMERS_SERVICE_URL", "http://localhost:8001/scrub_pii")

app = FastAPI(title="TRANSFORMERS Service", version="1.1.0")
instrumentator = Instrumentator().instrument(app).expose(app)
logger = logging.getLogger(__name__)

# -----------------------------
# TEXT CLEANING
# -----------------------------
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words("english"))

def convert_contact(text: str) -> str:
    try:
        response = requests.post(TRANSFORMERS_SERVICE_URL, json={"text": text}, timeout=3)
        response.raise_for_status()
        return response.json().get("scrubbed_text", text)
    except Exception as e:
        logger.warning(f"[PII Fallback] Transformers service failed: {e}")
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
        text = re.sub(r"\b(?:\+?\d{1,3})?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b", "[PHONE]", text)
        return text

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

# -----------------------------
# LOAD MODEL FROM MLFLOW
# -----------------------------
def load_transformer_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Connecting to MLflow: {MLFLOW_TRACKING_URI}")
    client = mlflow.MlflowClient()

    try:
        """exp = client.get_experiment_by_name("Transformer_Models")
        if not exp:
            raise ValueError("Experiment 'Transformer_Models' not found!")"""

        latest_versions = client.get_latest_versions(name=MODEL_NAME, stages=["None", "Staging", "Production"])
        if not latest_versions:
            raise ValueError(f"No versions found for model '{MODEL_NAME}'")

        version = latest_versions[0].version
        logger.info(f"Loading model '{MODEL_NAME}' version {version}")

        model_uri = f"models:/{MODEL_NAME}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded successfully: {MODEL_NAME} v{version}")
        return model, version

    except Exception as e:
        logger.error(f"Failed to load model '{MODEL_NAME}': {e}")
        raise RuntimeError(f"Could not load model: {e}")

# Load model at startup
model, latest_version = load_transformer_model()

# -----------------------------
# API MODELS
# -----------------------------
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str = "transformer"
    model_version: str

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    original_text = request.text.strip()
    if not original_text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    scrubbed_text = convert_contact(original_text)
    clean_text_str = clean_text(scrubbed_text)

    if not clean_text_str:
        raise HTTPException(status_code=400, detail="Input text is empty after cleaning")

    df = pd.DataFrame([clean_text_str], columns=["text"])
    predicted_label = model.predict(df)[0]

    # Extract probabilities
    try:
        hf_model = model._model_impl.python_model.model
        tokenizer = model._model_impl.python_model.tokenizer
        inputs = tokenizer(clean_text_str, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = hf_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        label_encoder = model._model_impl.python_model.label_encoder
        labels = label_encoder.classes_
        probabilities = {str(label): float(prob) for label, prob in zip(labels, probs)}
        confidence = float(probs.max())
    except Exception as e:
        logger.warning(f"Could not extract probabilities: {e}")
        probabilities = {}
        confidence = 1.0

    return PredictResponse(
        prediction=str(predicted_label),
        confidence=confidence,
        probabilities=probabilities,
        model_version=str(latest_version)
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "version": latest_version}

@app.get("/")
def root():
    return {
        "service": "TRANSFORMERS Service",
        "version": "1.1.0",
        "model": MODEL_NAME,
        "model_version": latest_version,
        "docs": "/docs"
    }

