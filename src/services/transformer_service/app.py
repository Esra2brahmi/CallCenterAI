import logging
import re
import string
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
import mlflow.pyfunc
import requests

# -------------------- CONFIG --------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
TRANSFORMERS_SERVICE_URL = "http://localhost:8001/scrub_pii"
TRANSFORMER_MODEL_NAME = "CallCenterAI_Transformer_Model"
TRANSFORMER_MODEL_VERSION = "2"
CLASSES = ['Hardware', 'HR Support', 'Access', 'Miscellaneous', 
           'Storage', 'Purchase', 'Internal Project', 'Administrative rights']

# -------------------- APP INIT --------------------
app = FastAPI(title="Transformer Service")
instrumentator = Instrumentator().instrument(app).expose(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- UTILITIES --------------------
def convert_contact(text: str) -> str:
    try:
        response = requests.post(TRANSFORMERS_SERVICE_URL, json={"text": text})
        response.raise_for_status()
        return response.json().get("scrubbed_text", text)
    except Exception as e:
        logger.warning(f"Transformer PII service failed: {e}. Using regex fallback.")
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b', '[PHONE]', text)
        return text

def clean_text(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------- MODEL LOADING --------------------
logger.info("Loading transformer model from MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{TRANSFORMER_MODEL_NAME}/{TRANSFORMER_MODEL_VERSION}")
logger.info("Model loaded successfully!")

# -------------------- SCHEMAS --------------------
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    probability: dict
    convertText: str

# -------------------- ENDPOINTS --------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text:
        logger.error("Text is required")
        raise HTTPException(status_code=400, detail="text is required!")

    try:
        converted_text = convert_contact(request.text)
        cleaned_text = clean_text(converted_text)
        
        # Predict using MLflow pyfunc model
        result = model.predict([cleaned_text])[0]  # expects dict: {"label": "Access", "probabilities": {...}}
        label = result['label']
        probabilities = result['probabilities']

        return PredictionResponse(label=label, probability=probabilities, convertText=converted_text)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="prediction failed!")

@app.get("/health")
async def health():
    return {"status": "healthy"}
