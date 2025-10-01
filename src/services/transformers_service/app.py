import os
from fastapi import FastAPI,HTTPException, logger
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel


app = FastAPI(title="Transformer service")

#monitor of the app : it collect all th metrics used by this fastAPI and it rturned into /metrics endpoint.
instrumentator = Instrumentator().instrument(app).expose(app)

# MLflow configuration
"""MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)"""

MODAL_NAME = "transformer_model.pkl" #to be added referenced to the model name trained

try:
    model_url = f"/models/{MODAL_NAME}"
    #in order to make the service adaptable with any modellogged in MLFlow
    model = mlflow.pyfunc.load_model(model_url)
    logger.info(f"transformer model has loaded successfully!")
except Exception as e:
    logger.error("failed to load transformer's model!")

class PredictionRequest(BaseModel):
    text:str

class PredictionResponse(BaseModel):
    label:str
    scores:dict

#Prediction end point
app.post("/predict",response_model=PredictionResponse)
def predict(tiket:PredictionRequest):
    if not tiket:
        logger.error(f"tiket shouldn't be empty!")
        raise HTTPException(status_code=400, details="tiket is be empty!")
    
    try:
        result = model.predict([tiket.text])[0]
        label = result['label']
        scores = {score['label']: score['scores'] for score in result['scores']}

        return PredictionResponse(label=label, scores=scores)
    except Exception as e:
        logger.error(f"prediction not found : {str(e)}!")
        raise HTTPException(status_code=500, detail="Prediction failed")

app.get("/health")
def health():
    return{"status":"healthy"}

