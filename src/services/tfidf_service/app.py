# file: app/main.py
import logging
import os
import re
import string
from typing import Dict

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from nltk.corpus import stopwords

# ==========================================================
# Configuration via variables d'environnement
# ==========================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "tfidf_svm_ticket_classifier")  # nom du modèle enregistré
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")                # ou "Staging", "None"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Ticket Classification API – TF-IDF + Calibrated LinearSVC",
    version="1.2.0",
    description="Prédit le Topic_group d’un ticket à partir de son texte",
)

Instrumentator().instrument(app).expose(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
logger.info(f"Chargement du modèle {MODEL_NAME} en stade {MODEL_STAGE}")


# ==========================================================
# Chargement du modèle (avec retry + fallback très clair)
# ==========================================================
"""def load_model_from_mlflow():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Tentative de chargement depuis MLflow → {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Modèle chargé avec succès depuis MLflow ({MODEL_NAME} v{model._model_meta.run_id[:8]}...)")
        return model
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow ne trouve pas le modèle {MODEL_NAME} en {MODEL_STAGE} : {e}")

        # Essai avec la dernière version "None" (cas fréquent après le 1er enregistrement)
        try:
            client = mlflow.MlflowClient()
            latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
            model_uri = f"models:/{MODEL_NAME}/{latest.version}"
            logger.info(f"Utilisation de la version {latest.version} (stage=None)")
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Modèle chargé depuis la dernière version disponible")
            return model
        except Exception:
            pass

    raise RuntimeError(f"Impossible de charger le modèle {MODEL_NAME} depuis MLflow")"""
# ==========================================================
# Chargement du modèle – version bulletproof 2025
# ==========================================================
def load_model_from_mlflow():
    logger.info(f"Recherche du modèle {MODEL_NAME} (stage préféré: {MODEL_STAGE})")

    client = mlflow.MlflowClient()

    # 1. Essai avec le stage demandé (Production, Staging, etc.)
    try:
        latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if latest:
            version = latest[0].version
            model_uri = f"models:/{MODEL_NAME}/{version}"
            logger.info(f"Chargement version {version} (stage {MODEL_STAGE})")
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.warning(f"Pas de version en {MODEL_STAGE} → {e}")

    # 2. Fallback : dernière version quel que soit le stage
    try:
        latest_any = client.get_latest_versions(MODEL_NAME, stages=None)  # None = toutes
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
except Exception as e:
    logger.critical(f"Échec critique du chargement du modèle : {e}")
    raise e


# ==========================================================
# Nettoyage du texte (identique à ton pipeline d’entraînement)
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
# Schemas
# ==========================================================
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Texte brut du ticket", example="My printer is not working anymore")


class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: Dict[str, float]
    cleaned_text: str
    model_version: str = None


# ==========================================================
# Routes
# ==========================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le champ 'text' ne peut pas être vide")

    cleaned = clean_text(request.text)

    try:
        probas = model.predict_proba([cleaned])[0]
        pred_idx = probas.argmax()
        predicted_label = model.classes_[pred_idx]

        prob_dict = {str(cls): float(p) for cls, p in zip(model.classes_, probas)}

        return PredictionResponse(
            predicted_label=str(predicted_label),
            probabilities=prob_dict,
            cleaned_text=cleaned,
            model_version=model._model_meta.run_id[:8] if hasattr(model, "_model_meta") else "unknown"
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
        "tracking_uri": MLFLOW_TRACKING_URI,
        "classes": list(model.classes_)
    }


@app.get("/")
def root():
    return {"message": "Ticket Classification API – TF-IDF + SVM", "docs": "/docs"}

"""import logging
import os
import re
import string
import socket
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import mlflow
from nltk.corpus import stopwords
from prometheus_fastapi_instrumentator import Instrumentator



MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"🚀 MLflow URI automatically selected: {MLFLOW_TRACKING_URI}")


# ==========================================================
# ⚙️ Config
# ==========================================================
MODEL_NAME = "tfidf_svm_model"
MODEL_STAGE = "Production"
LOCAL_MODEL_PATH = "models/tfidf_svm_model.pkl"

app = FastAPI(title="TF-IDF SVM Service", version="1.1.0")
Instrumentator().instrument(app).expose(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# 🧹 Text cleaning
# ==========================================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    stop_words = set(stopwords.words("english"))
    text = " ".join(w for w in text.split() if w not in stop_words)

    return text


# ==========================================================
# 📦 Load MLflow model with fallback
# ==========================================================
def load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Attempting to load MLflow model: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✅ MLflow model loaded successfully: {MODEL_NAME} ({MODEL_STAGE})")
        return model

    except Exception as e:
        logger.warning(f"⚠️ Could not load MLflow model: {e}")

        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"📦 Loading fallback local model: {LOCAL_MODEL_PATH}")
            return joblib.load(LOCAL_MODEL_PATH)

        raise RuntimeError("❌ No valid ML model available (MLflow failed, no local model).")


model = load_model()


# ==========================================================
# 🧾 API Schemas
# ==========================================================
class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    probability: dict
    cleaned: str


# ==========================================================
# 🔮 Prediction Endpoint
# ==========================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):

    if not req.text.strip():
        raise HTTPException(400, "Text is required")

    try:
        cleaned = clean_text(req.text)
        probs = model.predict_proba([cleaned])[0]
        label = model.classes_[probs.argmax()]
        probs_dict = {cls: float(p) for cls, p in zip(model.classes_, probs)}

        return PredictionResponse(
            label=label,
            probability=probs_dict,
            cleaned=cleaned
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction error: {e}")


# ==========================================================
# ❤️ Health Check
# ==========================================================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loaded_model": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI
    }
"""