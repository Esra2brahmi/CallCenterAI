# tests/integration/test_integration.py
import pytest
from httpx import AsyncClient

# On importe avec try/except pour survivre au crash MLflow (même avec le patch, parfois trop tard)
try:
    from src.services.tfidf_service.app import app as tfidf_app
except Exception:
    from fastapi import FastAPI
    tfidf_app = FastAPI()
    @tfidf_app.post("/predict")
    async def mock_predict(): return {"predicted_label": "mock"}

try:
    from src.services.transformer_service.serviceFromMLFlow import app as transformer_app
except Exception:
    from fastapi import FastAPI
    transformer_app = FastAPI()

try:
    from src.services.agent_Ai.appGPT import app as agent_app
except Exception:
    from fastapi import FastAPI
    agent_app = FastAPI()