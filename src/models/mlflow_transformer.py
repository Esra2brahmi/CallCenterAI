import os
import sys
import mlflow
import mlflow.pyfunc
import torch
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from mlflow.pyfunc import PythonModelContext

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models/transformer_model"))
RUN_NAME = "transformer_model_run"



print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] MODEL_DIR: {MODEL_DIR}")

# -----------------------------
# SAFETY CHECKS
# -----------------------------
required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "label_encoder.pkl",
]
missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))] 
if missing:
    print(f"[ERROR] Missing files in model directory: {missing}")
    sys.exit(1)

print("[DEBUG] All model files found ✅")

# -----------------------------
# DEFINE CUSTOM PYFUNC MODEL
# -----------------------------
class TransformerWrapper(mlflow.pyfunc.PythonModel):
    # ---------------------------------------
        #Charger un modèle MLflow en production
    # ---------------------------------------
    # j'ai remplaceé artifact_path=None (chemin obsolète) par load_context(self, context: PythonModelContext) (contienne des instruction qui vont automatiquement charger les artefacts depuis leurs emplacements MLflow corrects):
    def load_context(self, context: PythonModelContext):
        try:
            print("[DEBUG] Loading model artifacts inside pyfunc context...")
            model_path = context.artifacts["transformer_model"]   # coorectio: context.artifacts aulieu de artifacts_path if artifacts_path else self.artifacts["transformer_model"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
            self.model.eval()
            print("[DEBUG] Model + Tokenizer + LabelEncoder loaded successfully ✅")
        except Exception as e:
            print(f"[ERROR] Failed during load_context(): {e}")
            raise e

    def predict(self, context, model_input: pd.DataFrame):
        try:
            texts = model_input["text"].tolist()
            print(f"[DEBUG] Predict called with {len(texts)} texts")
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                #cpu: on met le modèle dans la mémoire du processeur (RAM). Il sert à le rendre utilisable et enregistrable sur n’importe quelle machine, même sans GPU.
                preds = torch.argmax(outputs.logits, dim=1).numpy()   #numpy to seul a un risque d’erreur si on a qu’un seul échantillon & risque de crash + incompatibilité avec MLflow donc on ajoute cpu
            labels = self.label_encoder.inverse_transform(preds)
            return labels.tolist()
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            raise e

# -----------------------------
# LOG MODEL TO MLFLOW (VERSION QUI MARCHE)
# -----------------------------
if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Transformer_Models")

        with mlflow.start_run(run_name=RUN_NAME) as run:
            print(f"[DEBUG] Run ID: {run.info.run_id}")

            # 1. Créer une instance et la charger manuellement pour infer la signature
            wrapper = TransformerWrapper()
            temp_context = PythonModelContext(
                artifacts={"transformer_model": MODEL_DIR},
                model_config={}
            )
            wrapper.load_context(temp_context)  # ← Charge vraiment le modèle

            # 2. Inférer la signature avec le modèle chargé
            example_input = pd.DataFrame({"text": ["example billing issue with payment"]})
            example_output = wrapper.predict(None, example_input)
            signature = mlflow.models.infer_signature(example_input, example_output)

            # 3. Optionnel : calcul des métriques (tu l’as déjà, c’est bon)

            # 4. LOG MODEL CORRECTEMENT
            mlflow.pyfunc.log_model(
                artifact_path="transformer_model",          
                python_model=wrapper,                        
                artifacts={"transformer_model": MODEL_DIR}, 
                input_example=example_input,
                signature=signature,
                registered_model_name="CallCenterTransformer",
                pip_requirements=[
                    "torch>=2.0",
                    "transformers>=4.30",
                    "scikit-learn",
                    "joblib",
                    "pandas",
                    "numpy"
                ],
                # code_paths=[os.path.dirname(__file__)],  # si tu as du code custom dans d'autres fichiers
            )

            print("[SUCCESS] Modèle loggé et chargé correctement dans MLflow")
            print(f"URI → models:/CallCenterTransformer/{run.info.run_id} ou via version")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
