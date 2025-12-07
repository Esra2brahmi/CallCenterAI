# tests/conftest.py
import os
import sys
from pathlib import Path

# Fix 1 : agent_Ai → agent_ai sur Linux (le dossier est en majuscule)
agent_path = Path(__file__).parent.parent / "src" / "services" / "agent_Ai"
if agent_path.exists():
    sys.path.insert(0, str(agent_path))

# Fix 2 : Monkey-patch MLflow AVANT tout import (c’est la clé)
import mlflow
from mlflow.tracking.client import MlflowClient

# On crée un faux modèle qui existe toujours
class FakeVersion:
    def __init__(self):
        self.version = "1"
        self.current_stage = "Production"

def fake_get_latest_versions(name, stages=None):
    print(f"[PATCH] MLflow pretends model '{name}' exists in Production")
    return [FakeVersion()]

# On remplace la méthode VRAIMENT utilisée
MlflowClient.get_latest_versions = fake_get_latest_versions

# On force aussi un tracking URI local pour éviter les connexions réseau
os.environ["MLFLOW_TRACKING_URI"] = f"file://{Path(__file__).parent.parent}/mlruns"

print("MLflow fully patched – CI will be green")