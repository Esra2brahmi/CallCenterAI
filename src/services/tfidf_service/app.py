import logging
import os
import re
import string
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import mlflow
import requests
from nltk.corpus import stopwords
from prometheus_fastapi_instrumentator import Instrumentator

# ==============================
# ⚙️ Configuration
# ==============================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "tfidf_svm_model")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/tfidf_svm_model.pkl")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
TRANSFORMERS_SERVICE_URL = os.getenv("TRANSFORMERS_SERVICE_URL", "http://localhost:8001/scrub_pii")

app = FastAPI(title="TF-IDF SVM Service", version="1.1.0")

# Monitoring Prometheus
instrumentator = Instrumentator().instrument(app).expose(app)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# 🧹 Prétraitement du texte
# ==============================
def convert_contact(text: str) -> str:
    """Supprime les données personnelles (PII) via microservice ou fallback regex."""
    try:
        response = requests.post(TRANSFORMERS_SERVICE_URL, json={"text": text}, timeout=3)
        response.raise_for_status()
        return response.json().get("scrubbed_text", text)
    except Exception as e:
        logger.warning(f"[Fallback] Échec appel Transformers : {e}")
        # Regex fallback
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
        text = re.sub(r"\b(?:\+?\d{1,3})?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b", "[PHONE]", text)
        return text

def clean_text(text: str) -> str:
    """Nettoyage du texte : lower, remove punctuation, digits, stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    stop_words = set(stopwords.words("english"))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ==============================
# 🔄 Chargement du modèle MLflow
# ==============================
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Connexion MLflow OK : {MLFLOW_TRACKING_URI}")
    client = mlflow.MlflowClient()
    
    try:
        # Récupère la dernière version loggée du modèle
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if latest_versions:
            latest_version = latest_versions[0].version
            logger.info(f"📦 Chargement du modèle '{MODEL_NAME}' version {latest_version} depuis MLflow")
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest_version}")
            return model
        else:
            raise ValueError(f"Aucune version trouvée pour le modèle '{MODEL_NAME}'")
    except Exception as e:
        logger.warning(f"[Fallback] Impossible de charger depuis MLflow : {e}")
        # Fallback : charger depuis le fichier local
        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Chargement du modèle depuis fichier local : {LOCAL_MODEL_PATH}")
            model = joblib.load(LOCAL_MODEL_PATH)
            return model
        else:
            logger.error("❌ Modèle introuvable ni dans MLflow ni localement.")
            raise RuntimeError("Impossible de charger le modèle TF-IDF")

model = load_model()

# ==============================
# 📦 Schémas de requête/réponse
# ==============================
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    probability: dict
    cleaned_text: str
    converted_text: str

# ==============================
# 🔮 Endpoint /predict
# ==============================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        converted_text = convert_contact(request.text)
        cleaned_text = clean_text(converted_text)

        class_probs = model.predict_proba([cleaned_text])[0]
        class_label = model.classes_[class_probs.argmax()]
        probs_dict = {cls: float(prob) for cls, prob in zip(model.classes_, class_probs)}

        logger.info(f"[Prediction] label={class_label}, probs={probs_dict}")

        return PredictionResponse(
            label=class_label,
            probability=probs_dict,
            cleaned_text=cleaned_text,
            converted_text=converted_text
        )

    except Exception as e:
        logger.exception("Erreur pendant la prédiction")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==============================
# ❤️ Health Check
# ==============================
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "stage": MODEL_STAGE}

# ==============================
# 🧭 Root
# ==============================
@app.get("/")
def home():
    return {"message": "Bienvenue sur le service TF-IDF + SVM 🚀", "version": "1.1.0"}
