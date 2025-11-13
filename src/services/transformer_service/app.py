from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from pydantic import BaseModel
import uvicorn
import logging
import time
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
import glob

# ==============================
# Logging setup
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Prometheus metrics
# ==============================
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    'transformer_requests_total',
    'Total number of requests to transformer service',
    registry=registry
)
REQUEST_LATENCY = Histogram(
    'transformer_request_latency_seconds',
    'Request latency in seconds',
    registry=registry
)
PREDICTION_COUNT = Counter(
    'transformer_predictions_total',
    'Total predictions made',
    ['predicted_class'],
    registry=registry
)

# ==============================
# FastAPI app
# ==============================
app = FastAPI(
    title="Transformer Service",
    description="Ticket classification using fine-tuned Transformer model",
    version="1.0.0"
)

# ==============================
# Globals
# ==============================
model = None
tokenizer = None
le = None  # Label encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mlartifacts/transformer_model"

# ==============================
# Schemas
# ==============================
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str = "transformer"

# ==============================
# Startup event
# ==============================
@app.on_event("startup")
async def load_model():
    global model, tokenizer, le

    try:
        logger.info("Loading transformer model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        # ✅ Load label encoder saved during training
        le = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")
        labels = list(le.classes_)

        # ✅ Update model.id2label to match training
        model.config.id2label = {i: label for i, label in enumerate(labels)}

        logger.info(f"Model loaded successfully on device: {device}")
        logger.info(f"Labels: {labels}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# ==============================
# Endpoints
# ==============================
@app.get("/label_mapping")
async def label_mapping():
    if model is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    mapping = {i: model.config.id2label[i] for i in range(model.config.num_labels)}
    return mapping

@app.get("/")
async def root():
    return {
        "service": "Transformer Service",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }

@app.get("/health")
async def health():
    if model is None or tokenizer is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": "loaded",
        "device": str(device),
        "classes": [model.config.id2label[i] for i in range(model.config.num_labels)],
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()
    REQUEST_COUNT.inc()

    if model is None or tokenizer is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        predicted_idx = torch.argmax(probs, dim=-1).item()
        predicted_label = le.inverse_transform([predicted_idx])[0]
        confidence = float(probs[0][predicted_idx])
        probabilities = {le.inverse_transform([i])[0]: float(probs[0][i]) for i in range(model.config.num_labels)}
        probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

        PREDICTION_COUNT.labels(predicted_class=predicted_label).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        return PredictResponse(
            prediction=predicted_label,
            confidence=confidence,
            probabilities=probabilities
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    REQUEST_COUNT.inc()
    if model is None or tokenizer is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    results = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        predicted_idx = torch.argmax(probs, dim=-1).item()
        predicted_label = le.inverse_transform([predicted_idx])[0]
        confidence = float(probs[0][predicted_idx])

        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": predicted_label,
            "confidence": confidence
        })

        PREDICTION_COUNT.labels(predicted_class=predicted_label).inc()

    return {"predictions": results}

@app.get("/metrics")
async def metrics():
    return JSONResponse(
        content=generate_latest(registry).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model_info")
async def model_info():
    if model is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "Transformer (HuggingFace)",
        "model_name": model.config.name_or_path if hasattr(model, "config") else "unknown",
        "num_classes": model.config.num_labels,
        "classes": [model.config.id2label[i] for i in range(model.config.num_labels)],
        "device": str(device),
        "max_length": 512
    }

# ==============================
# Run app
# ==============================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
