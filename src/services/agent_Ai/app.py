from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # Pour charger le modèle TF-IDF SVM
from transformers import pipeline  # Pour le modèle LLM
from langdetect import detect  # Pour détecter la langue
import re  # Pour scrub PII avec regex simples
from sklearn.feature_extraction.text import TfidfVectorizer  # Si besoin, mais assumons que le modèle est chargé avec vectorizer

app = FastAPI(
    title="AgentAI Service",
    description="Service intelligent pour router entre un modèle TF-IDF SVM et un modèle Transformers LLM, avec scrub PII et explication.",
    version="1.0.0"
)


tfidf_svm_model = joblib.load("tfidf_svm_model.pkl")  # Modèle TF-IDF + SVM
# Pour le vectorizer, si séparé, charger aussi
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Modèle LLM : utiliser un pipeline transformers, par exemple pour classification (adapter à votre tâche)
llm_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")  # Exemple, adapter à votre modèle

# Seuils exemples (à ajuster)
CONFIDENCE_THRESHOLD = 0.8  # Si confiance TF-IDF < seuil, router vers LLM
LENGTH_THRESHOLD = 100  # Si longueur > seuil, router vers LLM
SUPPORTED_LANGUAGES = ["en", "fr"]  # Langues supportées par TF-IDF, sinon LLM

# Fonction pour scrub PII (masquer emails, phones, noms potentiels - exemple simple avec regex)
def scrub_pii(text: str) -> str:
    # Masquer emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Masquer numéros de téléphone (exemple simple)
    text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})?[-. ]*(\d{4})\b', '[PHONE]', text)
    # Masquer numéros de carte (exemple)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)
    # Pour plus avancé, utiliser Presidio ou spaCy
    return text

# Modèle Pydantic pour la requête
class TextInput(BaseModel):
    text: str

# Prompt guidé pour LLM (adapter à votre tâche, ici exemple pour classification de sentiment)
def get_guided_prompt(text: str) -> str:
    return f"""
    Analysez le texte suivant et classifiez-le comme positif, négatif ou neutre.
    Fournissez une explication concise de votre raisonnement.
    Texte : {text}
    """

@app.post("/predict")
async def predict(input: TextInput):
    original_text = input.text
    if not original_text:
        raise HTTPException(status_code=400, detail="Texte vide")

    # Étape 1: Scrub PII
    scrubbed_text = scrub_pii(original_text)

    # Étape 2: Détecter langue
    try:
        language = detect(scrubbed_text)
    except:
        language = "unknown"

    # Étape 3: Calculer longueur
    text_length = len(scrubbed_text.split())  # Nombre de mots

    # Étape 4: Utiliser TF-IDF SVM pour prédiction initiale et confiance
    # Vectoriser le texte
    vectorized_text = tfidf_vectorizer.transform([scrubbed_text])
    prediction = tfidf_svm_model.predict(vectorized_text)[0]
    confidence = tfidf_svm_model.predict_proba(vectorized_text)[0].max()  # Confiance max (assumer SVM avec probas)

    # Étape 5: Routage intelligent
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

    # Étape 6: Choisir le modèle et prédire
    if use_llm:
        # Utiliser LLM avec prompt guidé
        guided_prompt = get_guided_prompt(scrubbed_text)
        llm_result = llm_pipeline(guided_prompt)[0]  # Adapter à votre sortie LLM
        result = {
            "model_used": "Transformers LLM",
            "prediction": llm_result['label'],
            "score": llm_result['score'],
            "explanation": " ".join(reasons) if reasons else "Routage par défaut vers LLM."
        }
    else:
        result = {
            "model_used": "TF-IDF SVM",
            "prediction": prediction,
            "confidence": confidence,
            "explanation": "Confiance haute, texte court, langue supportée."
        }

    # Ajouter le texte scrubbed
    result["scrubbed_text"] = scrubbed_text

    return result

# Pour lancer : uvicorn main:app --reload (assumer fichier nommé main.py)