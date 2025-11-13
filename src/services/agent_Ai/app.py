"""model distilgpt2
autre transformer pour repondre aux questions
et le transformer ancien pour classifier les tickets comme tfidf
sans if else dans le code"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from langdetect import detect
import re
import os
from typing import List, Dict

app = FastAPI(
    title="AgentAI Service",
    description="Service intelligent pour router entre un modèle TF-IDF SVM et ton propre modèle Transformers local, avec scrub PII et explication.",
    version="1.0.0"
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
        f"TF-IDF SVM model not found. Looked for: {TFIDF_SVM_CANDIDATES}.\n"
        "Place your model inside the `models/` folder or set MODELS_DIR environment variable."
    )
tfidf_svm_model = joblib.load(tfidf_svm_path)
print("Loaded TF-IDF SVM model:", tfidf_svm_model)

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
            # load label encoder
            _label_encoder = joblib.load(_LABEL_ENCODER_PATH)
            # update pipeline labels
            labels = list(_label_encoder.classes_)
            _llm_pipeline.model.config.id2label = {i: label for i, label in enumerate(labels)}
            print("✅ Transformer model loaded successfully with label encoder!")
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to initialize transformers pipeline '{_LLM_MODEL_PATH}': {e}"
            )
    return _llm_pipeline

# ==============================
# Thresholds & settings
# ==============================
CONFIDENCE_THRESHOLD = 0.75
LENGTH_THRESHOLD = 100
SUPPORTED_LANGUAGES = ["en", "fr"]

# ==============================
# PII scrub
# ==============================
def scrub_pii(text: str) -> str:
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})?[-. ]*(\d{4})\b', '[PHONE]', text)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)
    return text

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
def predict_with_llm(text: str, reasons: List[str] = None):
    llm = get_llm_pipeline()
    llm_result = llm(text)[0]

    # Safe label extraction
    raw_label = llm_result['label']
    try:
        # If label is LABEL_X format, map with encoder
        idx = int(raw_label.split("_")[1])
        predicted_label = _label_encoder.inverse_transform([idx])[0]
    except (IndexError, ValueError):
        # Otherwise, assume it's already the correct label string
        predicted_label = raw_label

    return {
        "model_used": "Your Local Transformer LLM",
        "prediction": predicted_label,
        "score": llm_result['score'],
        "explanation": " ".join(reasons) if reasons else "Routage par défaut vers LLM."
    }

def predict_with_tfidf(text: str):
    prediction = tfidf_svm_model.predict([text])[0]
    confidence = tfidf_svm_model.predict_proba([text])[0].max()
    return {
        "model_used": "TF-IDF SVM",
        "prediction": prediction,
        "confidence": confidence,
        "explanation": "Confiance haute, texte court, langue supportée."
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

    try:
        language = detect(scrubbed_text)
    except:
        language = "unknown"

    text_length = len(scrubbed_text.split())

    # TF-IDF SVM prediction
    prediction = tfidf_svm_model.predict([scrubbed_text])[0]
    confidence = tfidf_svm_model.predict_proba([scrubbed_text])[0].max()

    use_llm = False
    reasons = []

    if confidence < CONFIDENCE_THRESHOLD:
        use_llm = True
        reasons.append(f"Confiance TF-IDF SVM basse ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")
    if text_length > LENGTH_THRESHOLD:
        use_llm = True
        reasons.append(f"Texte long ({text_length} mots > {LENGTH_THRESHOLD})")
    if language not in SUPPORTED_LANGUAGES:
        use_llm = True
        reasons.append(f"Langue non supportée par TF-IDF ({language})")

    # Choose model
    if use_llm:
        result = predict_with_llm(scrubbed_text, reasons)
    else:
        result = predict_with_tfidf(scrubbed_text)

    result["scrubbed_text"] = scrubbed_text
    return result

@app.post("/batch_predict")
async def batch_predict(input: BatchTextInput):
    texts = input.texts
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    results = []
    for text in texts:
        scrubbed_text = scrub_pii(text)
        try:
            language = detect(scrubbed_text)
        except:
            language = "unknown"

        text_length = len(scrubbed_text.split())
        prediction = tfidf_svm_model.predict([scrubbed_text])[0]
        confidence = tfidf_svm_model.predict_proba([scrubbed_text])[0].max()

        use_llm = False
        reasons = []

        if confidence < CONFIDENCE_THRESHOLD:
            use_llm = True
            reasons.append(f"Confiance TF-IDF SVM basse ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")
        if text_length > LENGTH_THRESHOLD:
            use_llm = True
            reasons.append(f"Texte long ({text_length} mots > {LENGTH_THRESHOLD})")
        if language not in SUPPORTED_LANGUAGES:
            use_llm = True
            reasons.append(f"Langue non supportée par TF-IDF ({language})")

        if use_llm:
            result = predict_with_llm(scrubbed_text, reasons)
        else:
            result = predict_with_tfidf(scrubbed_text)

        result["scrubbed_text"] = scrubbed_text
        results.append(result)

    return {"predictions": results}
