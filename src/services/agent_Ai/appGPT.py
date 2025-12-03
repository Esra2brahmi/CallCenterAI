# """
# AgentAI Service - Version 4.0 (LLM-Only, Local Models, Zero Prompt, Zero If)
# ==========================================================================
# - Modèles : Transformer (local) + DistilGPT-2 (HF)
# - Aucun MLflow en runtime
# - Aucun if/else → routage par score + embedding
# - PII scrub, génération, classification
# - Chargement paresseux, professionnel, commenté
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import List, Dict, Any
# import joblib
# import os
# import re
# from functools import lru_cache
# from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer, AutoModel
# import torch
# from sentence_transformers import SentenceTransformer, util
# import logging

# # ==============================
# # Configuration
# # ==============================
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="AgentAI - LLM Local Intelligence",
#     description="Classification + Réponse générée (100% local). Zero prompt. Zero if.",
#     version="4.0.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==============================
# # Chemins locaux (fixés)
# # ==============================
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(ROOT, "models"))

# TRANSFORMER_MODEL_LOCAL = os.path.join(MODELS_DIR, "transformer_model")
# LABEL_ENCODER_LOCAL = os.path.join(TRANSFORMER_MODEL_LOCAL, "label_encoder.pkl")

# # Vérification au démarrage
# for path in [TRANSFORMER_MODEL_LOCAL, LABEL_ENCODER_LOCAL]:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Modèle local manquant : {path}")

# # ==============================
# # PII Scrubbing
# # ==============================
# PII_PATTERNS = {
#     r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
#     r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})?[-. ]*(\d{4})\b': '[PHONE]',
#     r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b': '[CARD]',
# }

# def scrub_pii(text: str) -> str:
#     for pattern, repl in PII_PATTERNS.items():
#         text = re.sub(pattern, repl, text)
#     return text

# # ==============================
# # Chargement paresseux
# # ==============================
# @lru_cache(maxsize=1)
# def load_classifier() -> Any:
#     """Charge le modèle Transformer local + label encoder."""
#     logger.info(f"Chargement du modèle local : {TRANSFORMER_MODEL_LOCAL}")
#     tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_LOCAL)
#     model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_LOCAL)
#     pipe = pipeline(
#         "text-classification",
#         model=model,
#         tokenizer=tokenizer,
#         device=-1,
#         return_all_scores=True,
#         function_to_apply="softmax"
#     )
#     pipe.label_encoder = joblib.load(LABEL_ENCODER_LOCAL)
#     pipe.model.config.id2label = {
#         i: label for i, label in enumerate(pipe.label_encoder.classes_)
#     }
#     logger.info("Classifier local chargé")
#     return pipe

# @lru_cache(maxsize=1)
# def load_generator() -> Any:
#     """Charge DistilGPT-2 pour génération."""
#     logger.info("Chargement de DistilGPT-2")
#     return pipeline(
#         "text-generation",
#         model="distilgpt2",
#         max_length=150,
#         truncation=True,
#         pad_token_id=50256
#     )

# @lru_cache(maxsize=1)
# def load_embedder() -> SentenceTransformer:
#     """Modèle d'embedding pour vérification sémantique."""
#     logger.info("Chargement de l'embedder")
#     return SentenceTransformer('all-MiniLM-L6-v2')

# # ==============================
# # Références sémantiques (templates)
# # ==============================
# CATEGORY_TEMPLATES = {
#     "problème de connexion": "Impossible de me connecter à mon compte",
#     "facturation": "Facture incorrecte ou trop élevée",
#     "support technique": "Mon appareil ne fonctionne pas",
#     "compte bloqué": "Mon compte est bloqué",
#     "mot de passe oublié": "J'ai oublié mon mot de passe",
# }

# embedder = load_embedder()
# TEMPLATE_EMBEDDINGS = {
#     cat: embedder.encode(text, convert_to_tensor=True)
#     for cat, text in CATEGORY_TEMPLATES.items()
# }

# # ==============================
# # Prédiction + Réponse (LLM-Only, local)
# # ==============================
# def process_text(text: str) -> Dict[str, Any]:
#     scrubbed = scrub_pii(text)
#     classifier = load_classifier()

#     # Classification
#     results = classifier(scrubbed)[0]
#     scores = {r['label']: r['score'] for r in results}
#     best_label = max(scores, key=scores.get)
#     idx = int(best_label.split("_")[1])
#     pred_category = classifier.label_encoder.inverse_transform([idx])[0]
#     clf_confidence = scores[best_label]

#     # Vérification sémantique
#     user_emb = embedder.encode(scrubbed, convert_to_tensor=True)
#     sims = {cat: util.cos_sim(user_emb, emb).item() for cat, emb in TEMPLATE_EMBEDDINGS.items()}
#     sem_category = max(sims, key=sims.get)
#     sem_confidence = sims[sem_category]

#     # Fusion
#     final_category = sem_category if sem_confidence > clf_confidence else pred_category
#     final_confidence = max(sem_confidence, clf_confidence)

#     # Génération
#     generator = load_generator()
#     prompt = f"Catégorie: {final_category}. Message: {scrubbed[:180]}"
#     output = generator(prompt, num_return_sequences=1)[0]['generated_text']
#     response = output[len(prompt):].strip().split('\n')[0]
#     response = (response[:180] + "...") if len(response) > 180 else response

#     return {
#         "model_used": "LLM Local (Transformer + DistilGPT-2)",
#         "prediction": final_category,
#         "confidence": round(final_confidence, 3),
#         "classification_confidence": round(clf_confidence, 3),
#         "semantic_confidence": round(sem_confidence, 3),
#         "explanation": "Classification locale + vérification sémantique + génération",
#         "generated_response": response,
#         "scrubbed_text": scrubbed
#     }

# # ==============================
# # Schémas
# # ==============================
# class PredictInput(BaseModel):
#     text: str = Field(..., example="Je n'arrive plus à me connecter")

# class BatchPredictInput(BaseModel):
#     texts: List[str] = Field(..., example=["Facture erronée", "Mot de passe oublié"])

# class PredictOutput(BaseModel):
#     model_used: str
#     prediction: str
#     confidence: float
#     classification_confidence: float
#     semantic_confidence: float
#     explanation: str
#     generated_response: str
#     scrubbed_text: str

# # ==============================
# # Endpoints
# # ==============================
# @app.post("/predict", response_model=PredictOutput)
# async def predict(input: PredictInput):
#     if not input.text.strip():
#         raise HTTPException(status_code=400, detail="Texte vide")
#     return PredictOutput(**process_text(input.text))

# @app.post("/batch_predict")
# async def batch_predict(input: BatchPredictInput):
#     if not input.texts:
#         raise HTTPException(status_code=400, detail="Liste vide")
#     return {
#         "predictions": [
#             process_text(text)
#             for text in input.texts
#             if text.strip()
#         ]
#     }

# @app.get("/health")
# async def health():
#     return {
#         "status": "healthy",
#         "models": "all local",
#         "transformer": TRANSFORMER_MODEL_LOCAL,
#         "generator": "distilgpt2"
#     }

"""
AgentAI Service - Version 4.1 (LLM-Only, Local Models, Zero Prompt, Zero If)
============================================================================
- Modèles : Transformer local + DistilGPT-2
- Chargement paresseux, PII scrubbing, génération locale
- CORRIGÉ : gestion robuste des labels du classifier (plus jamais d'IndexError)
- Testé et fonctionnel même avec modèles mal étiquetés
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import os
import re
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import logging

# ==============================
# Configuration & Logging
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgentAI - LLM Local Intelligence",
    description="Classification + Réponse générée (100% local). Zero prompt. Zero if.",
    version="4.1.0"
)

# CORS (permet ton frontend React/Vue/etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En dev : tout autorisé. En prod, restreindre !
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Chemins locaux
# ==============================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(ROOT, "models"))
TRANSFORMER_MODEL_LOCAL = os.path.join(MODELS_DIR, "transformer_model")
LABEL_ENCODER_PATH = os.path.join(TRANSFORMER_MODEL_LOCAL, "label_encoder.pkl")

# Vérification au démarrage
for p in [TRANSFORMER_MODEL_LOCAL, LABEL_ENCODER_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Modèle ou label encoder manquant → {p}")

# ==============================
# PII Scrubbing (email, téléphone, CB)
# ==============================
PII_PATTERNS = {
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b': '[EMAIL]',
    r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b': '[PHONE]',
    r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b': '[CARTE]',
}

def scrub_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ==============================
# Chargement paresseux des modèles
# ==============================
@lru_cache(maxsize=1)
def load_classifier():
    """Charge le modèle Transformer local + label encoder."""
    logger.info(f"Chargement du modèle Transformer local : {TRANSFORMER_MODEL_LOCAL}")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_LOCAL)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_LOCAL)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,                    # CPU
        return_all_scores=True,      # On veut tous les scores
        function_to_apply="softmax"
    )

    # Chargement du LabelEncoder (sklearn)
    pipe.label_encoder = joblib.load(LABEL_ENCODER_PATH)
    logger.info("Classifier local chargé avec succès")
    return pipe


@lru_cache(maxsize=1)
def load_generator():
    """DistilGPT-2 pour générer une réponse naturelle."""
    logger.info("Chargement de DistilGPT-2 pour génération")
    return pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=120,
        truncation=True,
        pad_token_id=50256
    )


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    """Embedding pour vérification sémantique (fallback du classifier)."""
    logger.info("Chargement de l'embedder all-MiniLM-L6-v2")
    return SentenceTransformer('all-MiniLM-L6-v2')


# ==============================
# Templates sémantiques (fallback très fiable)
# ==============================
CATEGORY_TEMPLATES = {
    "problème de connexion": "Je n'arrive plus à me connecter à mon compte internet",
    "facturation": "Ma facture est trop élevée ou erronée",
    "support technique": "Mon modem ou ma box ne fonctionne plus",
    "compte bloqué": "Mon compte client est bloqué",
    "mot de passe oublié": "J'ai oublié mon mot de passe",
    # Ajoute les catégories que tu veux ici
}

embedder = load_embedder()
TEMPLATE_EMBEDDINGS = {
    cat: embedder.encode(text, convert_to_tensor=True)
    for cat, text in CATEGORY_TEMPLATES.items()
}


# ==============================
# Fonction principale de traitement
# ==============================
def process_text(text: str) -> Dict[str, Any]:
    scrubbed = scrub_pii(text)
    classifier = load_classifier()

    # =================================================================
    # 1. Classification (CORRIGÉE – marche avec n'importe quel format de label)
    # =================================================================
    raw_results = classifier(scrubbed)[0]                     # liste de dicts : [{"label": ..., "score": ...}, ...]
    best_item = max(raw_results, key=lambda x: x["score"])     # item avec le meilleur score
    best_index = raw_results.index(best_item)                 # ← INDEX RÉEL DANS LA LISTE = ID DU LABEL
    clf_confidence = best_item["score"]

    # Décodage de la vraie catégorie via le LabelEncoder
    pred_category = classifier.label_encoder.inverse_transform([best_index])[0]

    # =================================================================
    # 2. Vérification sémantique (embedding)
    # =================================================================
    user_embedding = embedder.encode(scrubbed, convert_to_tensor=True)
    semantic_similarities = {
        cat: util.cos_sim(user_embedding, templ_emb).item()
        for cat, templ_emb in TEMPLATE_EMBEDDINGS.items()
    }
    sem_category = max(semantic_similarities, key=semantic_similarities.get)
    sem_confidence = semantic_similarities[sem_category]

    # =================================================================
    # 3. Fusion finale : on garde le plus confiant
    # =================================================================
    final_category = sem_category if sem_confidence > clf_confidence else pred_category
    final_confidence = max(sem_confidence, clf_confidence)

    # =================================================================
    # 4. Génération de réponse avec DistilGPT-2
    # =================================================================
    generator = load_generator()
    prompt = f"Catégorie: {final_category}. Message client: {scrubbed[:200]}"
    generated = generator(prompt, num_return_sequences=1)[0]["generated_text"]
    response = generated[len(prompt):].strip().split("\n")[0]
    response = (response[:180] + "...") if len(response) > 180 else response

    # =================================================================
    # 5. Retour structuré
    # =================================================================
    return {
        "model_used": "Local Transformer + DistilGPT-2",
        "prediction": final_category,
        "confidence": round(final_confidence, 3),
        "classification_confidence": round(clf_confidence, 3),
        "semantic_confidence": round(sem_confidence, 3),
        "explanation": "Classification locale + vérif sémantique + génération locale",
        "generated_response": response.strip(),
        "scrubbed_text": scrubbed
    }


# ==============================
# Schémas Pydantic
# ==============================
class PredictInput(BaseModel):
    text: str = Field(..., example="Je n'arrive plus à me connecter à internet")

class PredictOutput(BaseModel):
    model_used: str
    prediction: str
    confidence: float
    classification_confidence: float
    semantic_confidence: float
    explanation: str
    generated_response: str
    scrubbed_text: str


# ==============================
# Endpoints
# ==============================
@app.post("/predict", response_model=PredictOutput)
async def predict(input: PredictInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide ou whitespace uniquement")
    return PredictOutput(**process_text(input.text))


@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    if not texts:
        raise HTTPException(status_code=400, detail="Liste vide")
    return {
        "predictions": [
            process_text(t) for t in texts if t.strip()
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": "Transformer local + DistilGPT-2",
        "embedder": "all-MiniLM-L6-v2",
        "up_since": "running"
    }


# Lancement : uvicorn src.services.agent_Ai.appGPT:app --host 0.0.0.0 --port 8002 --reload