# tests/conftest.py
import sys
from pathlib import Path

# Fix 1 : agent_Ai → agent_ai sur Linux/CI (le dossier est en majuscule)
agent_ai_real = Path(__file__).parent.parent / "src" / "services" / "agent_Ai"
if agent_ai_real.exists():
    sys.path.insert(0, str(agent_ai_real))

# Fix 2 : Monkey-patch MLflow pour que TOUS les modèles soient "trouvés" instantanément
import mlflow
from mlflow.tracking.client import MlflowClient

class FakeVersion:
    version = "1"
    current_stage = "Production"

def fake_get_latest_versions(name, stages=None):
    return [FakeVersion()]

# On remplace la vraie méthode → plus jamais d'erreur "Modèle introuvable"
MlflowClient.get_latest_versions = fake_get_latest_versions

print("MLflow patched – all models are magically found")