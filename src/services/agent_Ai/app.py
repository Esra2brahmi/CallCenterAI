from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import os
from typing import List, Dict
import torch

app = FastAPI(
    title="AgentAI Service with LLM Router",
    description="Service with DistilGPT-2 router for intelligent model selection",
    version="2.0.0"
)

# ==============================
# Paths
# ==============================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(ROOT, "models"))

TFIDF_SVM_CANDIDATES = [
    os.path.join(MODELS_DIR, "tfidf_svm_model.pkl"),
    os.path.join(MODELS_DIR, "tf_idf_svm_model.pkl"),
    os.path.join(MODELS_DIR, "svm_model.pkl"),
]

def _find_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ==============================
# Load TF-IDF + SVM model
# ==============================
tfidf_svm_path = _find_first_existing(TFIDF_SVM_CANDIDATES)
if not tfidf_svm_path:
    raise FileNotFoundError(
        f"TF-IDF SVM model not found. Looked for: {TFIDF_SVM_CANDIDATES}"
    )
tfidf_svm_model = joblib.load(tfidf_svm_path)
print("✅ Loaded TF-IDF SVM model")

# ==============================
# Load DistilGPT-2 Router Model
# ==============================
_router_model = None
_router_tokenizer = None
_ROUTER_MODEL_PATH = os.path.join(MODELS_DIR, "distilgpt2_router")

def get_router():
    """Lazy load router model"""
    global _router_model, _router_tokenizer
    if _router_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print(f"Loading DistilGPT-2 Router from: {_ROUTER_MODEL_PATH}")
        _router_tokenizer = AutoTokenizer.from_pretrained(_ROUTER_MODEL_PATH)
        _router_model = AutoModelForSequenceClassification.from_pretrained(_ROUTER_MODEL_PATH)
        _router_model.eval()  # Set to evaluation mode
        
        print("✅ Router model loaded successfully!")
    return _router_model, _router_tokenizer

# ==============================
# Lazy Transformer LLM setup
# ==============================
_llm_pipeline = None
_label_encoder = None

_LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_NAME",
    os.path.join(MODELS_DIR, "transformer_model")
)
_LABEL_ENCODER_PATH = os.path.join(_LLM_MODEL_PATH, "label_encoder.pkl")

def get_llm_pipeline():
    global _llm_pipeline, _label_encoder
    if _llm_pipeline is None:
        from transformers import pipeline as _pipeline
        try:
            print(f"Loading Transformer model from: {_LLM_MODEL_PATH}")
            _llm_pipeline = _pipeline(
                "text-classification",
                model=_LLM_MODEL_PATH,
                tokenizer=_LLM_MODEL_PATH,
                device=-1  # CPU
            )
            _label_encoder = joblib.load(_LABEL_ENCODER_PATH)
            labels = list(_label_encoder.classes_)
            _llm_pipeline.model.config.id2label = {i: label for i, label in enumerate(labels)}
            print("✅ Transformer model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load transformer: {e}")
    return _llm_pipeline

# ==============================
# PII scrub
# ==============================
def scrub_pii(text: str) -> str:
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})?[-. ]*(\d{4})\b', '[PHONE]', text)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)
    return text

# ==============================
# Router Logic
# ==============================
def predict_routing(text: str) -> Dict:
    """
    Use DistilGPT-2 to predict which model to use
    Returns: dict with 'use_transformer' (bool) and 'confidence' (float)
    """
    router_model, router_tokenizer = get_router()
    
    # Tokenize
    inputs = router_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = router_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    
    # 0 = TF-IDF, 1 = Transformer
    use_transformer = (prediction == 1)
    
    return {
        "use_transformer": use_transformer,
        "confidence": confidence,
        "router_prediction": "Transformer" if use_transformer else "TF-IDF"
    }

# ==============================
# Schemas
# ==============================
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

# ==============================
# Prediction helpers
# ==============================
def predict_with_llm(text: str, routing_info: Dict):
    llm = get_llm_pipeline()
    llm_result = llm(text)[0]

    # Safe label extraction
    raw_label = llm_result['label']
    try:
        idx = int(raw_label.split("_")[1])
        predicted_label = _label_encoder.inverse_transform([idx])[0]
    except (IndexError, ValueError):
        predicted_label = raw_label

    return {
        "model_used": "Transformer (Local LLM)",
        "prediction": predicted_label,
        "score": llm_result['score'],
        "router_confidence": routing_info["confidence"],
        "explanation": f"Router selected Transformer with {routing_info['confidence']:.2%} confidence"
    }

def predict_with_tfidf(text: str, routing_info: Dict):
    prediction = tfidf_svm_model.predict([text])[0]
    confidence = tfidf_svm_model.predict_proba([text])[0].max()
    return {
        "model_used": "TF-IDF SVM",
        "prediction": prediction,
        "confidence": confidence,
        "router_confidence": routing_info["confidence"],
        "explanation": f"Router selected TF-IDF with {routing_info['confidence']:.2%} confidence"
    }

# ==============================
# Endpoints
# ==============================
@app.post("/predict")
async def predict(input: TextInput):
    original_text = input.text
    if not original_text:
        raise HTTPException(status_code=400, detail="Texte vide")

    scrubbed_text = scrub_pii(original_text)

    # Use DistilGPT-2 router to decide which model to use
    routing_decision = predict_routing(scrubbed_text)
    
    # Route to appropriate model
    if routing_decision["use_transformer"]:
        result = predict_with_llm(scrubbed_text, routing_decision)
    else:
        result = predict_with_tfidf(scrubbed_text, routing_decision)

    result["scrubbed_text"] = scrubbed_text
    result["routing_decision"] = routing_decision
    
    return result

@app.post("/batch_predict")
async def batch_predict(input: BatchTextInput):
    texts = input.texts
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    results = []
    for text in texts:
        scrubbed_text = scrub_pii(text)
        
        # Router decision
        routing_decision = predict_routing(scrubbed_text)
        
        # Route to model
        if routing_decision["use_transformer"]:
            result = predict_with_llm(scrubbed_text, routing_decision)
        else:
            result = predict_with_tfidf(scrubbed_text, routing_decision)

        result["scrubbed_text"] = scrubbed_text
        result["routing_decision"] = routing_decision
        results.append(result)

    return {"predictions": results}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tfidf_loaded": tfidf_svm_model is not None,
        "router_loaded": _router_model is not None,
        "llm_loaded": _llm_pipeline is not None
    }

@app.get("/router_info")
async def router_info():
    """Get information about the router model"""
    router_model, _ = get_router()
    return {
        "router_type": "DistilGPT-2",
        "model_path": _ROUTER_MODEL_PATH,
        "num_parameters": sum(p.numel() for p in router_model.parameters()),
        "labels": {
            0: "TF-IDF SVM",
            1: "Transformer LLM"
        }
    }