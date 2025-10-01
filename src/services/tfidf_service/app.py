from fastapi import FastAPI, HTTPException, logger
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel



app = FastAPI(title="TF-IDF SVM service")

#monitor of the app : it collect all th metrics used by this fastAPI and it rturned into /metrics endpoint.
instrumentator = Instrumentator().instrument(app).expose(app)

# MLflow configuration
"""MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)"""

""" Load model trained"""
try:
    model = mlflow.sklearn.load_model("models:/tfidf_svm_model/Production")
    logger.info("TF-IDF model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model from MLflow")


class PredictionRequest(BaseModel):
    text:str

class PredictionResponse(BaseModel):
    label:str
    probability:dict

"""Prediction end point"""
@app.post("/predict", response_model=PredictionResponse)
def predict(request:PredictionRequest):
    if not request.text :
        logger.error("text is required")
        raise HTTPException(status=400, detail="text is required!")
    
    try:
        # text_exemple: I love this product!
        #predict class probabilities
        class_probability = model.predict_proba([request.text])[0]   #probs = [0.2, 0.8]
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
        return PredictionResponse(label=class_label,probability=probabilities)
    except Exception as e:
        logger.error(f"prediction error: {str (e)}")
        raise HTTPException(status_code=500, detail="prediction failed!")
    

    

@app.get("/health")
async def health():
    return {"status": "healthy"}