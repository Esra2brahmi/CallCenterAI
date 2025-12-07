from unittest.mock import patch
from src.services.agent_Ai.appGPT import classify_and_respond

@patch("src.services.agent_Ai.agent.requests.post")
def test_agent_mocked(mock_post):
    mock_post.return_value.json.return_value = {"prediction": "mot de passe oublié"}
    result = classify_and_respond("J'ai oublié mon mot de passe")
    assert "mot de passe" in result.lower()