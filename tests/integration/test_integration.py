import pytest
from httpx import AsyncClient
from src.services.tfidf_service.app import app as tfidf_app
from src.services.transformer_service.serviceFromMLFlow import app as transformer_app
from src.services.agent_Ai.appGPT import app as agent_app

import os
os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")  # CI fix

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tfidf_full_flow():
    async with AsyncClient(app=tfidf_app, base_url="http://test") as client:
        resp = await client.post("/predict", json={"text": "printer not working"})
        assert resp.status_code == 200
        assert resp.json()["prediction"] is not None

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_calls_services():
    async with AsyncClient(app=agent_app, base_url="http://test") as client:
        resp = await client.post("/chat", json={"message": "I forgot my password"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] or data["response"]