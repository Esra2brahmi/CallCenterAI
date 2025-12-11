# log_transformer_model.py
import os
import sys
import mlflow
import mlflow.pyfunc
import torch
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mlflow.models import infer_signature
from mlflow.types.schema import Schema, ColSpec

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models/transformer_model"))
RUN_NAME = "transformer_model_run_v2"
MODEL_NAME = "CallCenterTransformer"


required_files = [
    "config.json",
    "model.safetensors",  # ou pytorch_model.bin
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "label_encoder.pkl",
]

missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing:
    print(f"[ERROR] Fichiers manquants dans {MODEL_DIR} : {missing}")
    sys.exit(1)

print("[DEBUG] Tous les fichiers du modèle sont présents")

# -----------------------------
# CUSTOM PYFUNC WRAPPER (FIXED & ROBUST)
# -----------------------------
class TransformerWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        print("[DEBUG] Chargement des artefacts via MLflow context...")
        
        # Chemin vers le dossier du modèle Hugging Face
        model_path = context.artifacts["transformer_model"]

        # FORCE LEGACY ATTENTION → Évite l'erreur DistilBertSdpaAttention en prod !
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # CLÉ MAGIQUE : désactive SDPA (né en 4.36+
        )
        
        # Charger le label encoder
        self.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
        
        self.model.eval()
        print("[SUCCESS] Modèle, tokenizer et label_encoder chargés avec succès")

    def predict(self, context, model_input):
        # Supporte DataFrame ou list/str
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            raise ValueError("Input must be str, list or DataFrame with 'text' column")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        labels = self.label_encoder.inverse_transform(preds)
        return labels.tolist()


# -----------------------------
# LOGGING DU MODÈLE (VERSION QUI MARCHE À 100% EN PROD)
# -----------------------------
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Transformer_Models")

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f"[INFO] Run ID: {run.info.run_id}")

        # 1. Créer et charger le wrapper temporairement pour infer la signature
        wrapper = TransformerWrapper()
        dummy_context = mlflow.pyfunc.PythonModelContext(
            artifacts={"transformer_model": MODEL_DIR},
            model_config={}
        )
        wrapper.load_context(dummy_context)

        # 2. Exemple d'entrée/sortie pour la signature
        example_input = pd.DataFrame({"text": ["my bill is wrong", "i want to cancel"]})
        example_output = wrapper.predict(None, example_input)

        # 3. Inférer la signature proprement
        signature = infer_signature(example_input, example_output)

        # 4. LOG avec les bons paramètres
        mlflow.pyfunc.log_model(
            artifact_path="transformer_model",
            python_model=wrapper,
            artifacts={"transformer_model": MODEL_DIR},
            signature=signature,
            input_example=example_input,
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "torch==2.3.1",                   # Pin exact pour stabilité
                "transformers==4.42.4",            # Version qui a créé le modèle
                "scikit-learn>=1.3",
                "joblib>=1.2",
                "pandas",
                "numpy"
            ],
            metadata={
                "model_type": "distilbert",
                "task": "text-classification",
                "max_length": 128
            }
        )

        print(f"[SUCCESS] Modèle enregistré avec succès !")
        print(f"→ Utilise : models:/{MODEL_NAME}/latest")
        print(f"→ Ou version spécifique : models:/{MODEL_NAME}/{run.info.run_id}")