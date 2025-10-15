import logging
import os
import re
import string
from fastapi import FastAPI, HTTPException, logger, requests
import joblib
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from nltk.corpus import stopwords


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TRANSFORMERS_SERVICE_URL = os.getenv("TRANSFORMERS_SERVICE_URL", "http://localhost:8001/scrub_pii")

app = FastAPI(title="TF-IDF SVM service")

#monitor of the app : it collect all th metrics used by this fastAPI and it rturned into /metrics endpoint.
instrumentator = Instrumentator().instrument(app).expose(app)

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Remove PII (emails, phone numbers) from text.
def convertContact(text: str) -> str:
    try:
        response = requests.post(TRANSFORMERS_SERVICE_URL, json={"text": text})
        response.raise_for_status()
        return response.json()["scrubbed_text"]
    except Exception as e:
        logger.warning(f"Échec appel transformers: {e}. Fallback sur regex.")
        # Regex fallback (votre code original)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b', '[PHONE]', text)
        return text

def cleanText(text:str)->str:
    stopWords = set(stopwords.words("english"))
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in stopWords]
    return " ".join(words)

def loadModel():
    """ Load model trained"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model("models:/tfidf_svm_model/Production")
        #model = mlflow.pyfunc.load_model("models:/tfidf_svm_model/Production")
        logger.info("TF-IDF model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model from MLflow")
    
    

model = loadModel()


class PredictionRequest(BaseModel):
    text:str

class PredictionResponse(BaseModel):
    label:str
    probability:dict
    convertText: str

"""Prediction end point"""
@app.post("/predict", response_model=PredictionResponse)
def predict(request:PredictionRequest):
    if not request.text :
        logger.error("text is required")
        raise HTTPException(status=400, detail="text is required!")
    
    try:
        #convert contact and clean texts:
        convertedText = convertContact(request.text)
        cleanedText = cleanText(convertedText)
        # text_exemple: I love this product!
        #predict class probabilities
        class_probability = model.predict_proba([cleanedText])[0]   #probs = [0.2, 0.8]
        print(f"{class_probability}")
        #Determines the predicted label by selecting the class with the highest probability.
        #classes = ['negative', 'positive']
        class_label = model.classes_[class_probability.argmax()] 
        print(f"{class_label}")
        #Converts probabilities to a dictionary mapping class names to their probabilities.
        probabilities = {cls: float(prob) for cls, prob in zip(model.classes_, class_probability)}  #probabilities = {"negative": 0.2, "positive": 0.8}
        """
        result: { "label": "positive", "probabilities": {"negative": 0.2,"positive": 0.8}}
        """
        return PredictionResponse(
            label=class_label,
            probability=probabilities,
            convertText=convertedText
        )
    except Exception as e:
        logger.error(f"prediction error: {str (e)}")
        raise HTTPException(status_code=500, detail="prediction failed!")
    

    

@app.get("/health")
async def health():
    return {"status": "healthy"}


"""pipeline = joblib.load("models/tf_idf_svm_model.pkl")
test_text = "mail please dear looks blacklisted receiving mails anymore sample attached thanks kind regards senior engineer" 
prediction = pipeline.predict([test_text])
probs = pipeline.predict_proba([test_text])[0]
print(f"Predicted label: {prediction[0]}")
print(f"Probabilities: {probs}")"""