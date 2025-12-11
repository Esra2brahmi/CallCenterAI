import logging
import os
import re
import string
import time
from typing import Dict

import joblib
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from nltk.corpus import stopwords
from prometheus_client import Counter, Histogram



PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Nombre total de prédictions effectuées",
    ["service", "model", "predicted_label"]
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Latence de la prédiction uniquement (hors FastAPI overhead)",
    ["service", "model"]
)


# ==========================================================
# FORCER MLFLOW À UTILISER DES CHEMINS RELATIFS (LA LIGNE MAGIQUE)
# ==========================================================
# Cette ligne résout TOUS les problèmes Windows + Docker + MLflow
os.environ["MLFLOW_ARTIFACT_URI"] = f"file:{os.getcwd()}/mlruns"

# ==========================================================
# Configuration via variables d'environnement
# ==========================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "tfidf_svm_ticket_classifier")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Ticket Classification API – TF-IDF + Calibrated LinearSVC",
    version="1.2.1",
    description="Prédit le Topic_group d’un ticket à partir de son texte",
)

Instrumentator().instrument(app).expose(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
logger.info(f"Chargement du modèle {MODEL_NAME} en stade {MODEL_STAGE}")

# ==========================================================
# Chargement du modèle – version bulletproof 2025
# ==========================================================

def load_model_from_mlflow():
    logger.info(f"Recherche du modèle {MODEL_NAME} (stage préféré: {MODEL_STAGE})")

    client = mlflow.MlflowClient()

    try:
        latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if latest:
            version = latest[0].version
            model_uri = f"models:/{MODEL_NAME}/{version}"
            logger.info(f"Chargement version {version} (stage {MODEL_STAGE})")
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.warning(f"Pas de version en {MODEL_STAGE} → {e}")

    try:
        latest_any = client.get_latest_versions(MODEL_NAME, stages=None)
        if latest_any:
            version = latest_any[0].version
            model_uri = f"models:/{MODEL_NAME}/{version}"
            logger.info(f"Fallback → version {version} (stage {latest_any[0].current_stage})")
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Aucune version trouvée pour {MODEL_NAME} → {e}")

    raise RuntimeError(f"Modèle {MODEL_NAME} introuvable dans le Model Registry")


# Chargement au démarrage
try:
    model = load_model_from_mlflow()
    logger.info("Modèle chargé avec succès !")
except Exception as e:
    logger.critical(f"Échec critique du chargement du modèle : {e}")
    raise e


# ==========================================================
# Nettoyage du texte
# ==========================================================
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


# ==========================================================
# Schemas & Routes (inchangées)
# ==========================================================
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Texte brut du ticket", example="My printer is not working anymore")

class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: Dict[str, float]
    cleaned_text: str
    model_version: str = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le champ 'text' ne peut pas être vide")

    cleaned = clean_text(request.text)
    start = time.time()

    try:
        probas = model.predict_proba([cleaned])[0]
        pred_idx = probas.argmax()
        predicted_label = model.classes_[pred_idx]

        # MÉTRIQUES CUSTOM
        PREDICTIONS_TOTAL.labels(service="tfidf_svc", model=MODEL_NAME, predicted_label=predicted_label).inc()
        PREDICTION_LATENCY.labels(service="tfidf_svc", model=MODEL_NAME).observe(time.time() - start)

        prob_dict = {str(cls): float(p) for cls, p in zip(model.classes_, probas)}

        return PredictionResponse(
            predicted_label=str(predicted_label),
            probabilities=prob_dict,
            cleaned_text=cleaned,
            model_version=getattr(model, "run_id", "unknown")[:8]
        )
    except Exception as e:
        logger.exception("Erreur pendant la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne du modèle")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE,
        "classes": list(model.classes_)
    }

@app.get("/")
def root():
    return {"message": "TF-IDF + SVM Ticket Classifier", "docs": "/docs"}