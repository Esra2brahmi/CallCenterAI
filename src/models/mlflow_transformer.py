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
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=128
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                #cpu: on met le modèle dans la mémoire du processeur (RAM). Il sert à le rendre utilisable et enregistrable sur n’importe quelle machine, même sans GPU.
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()   #numpy to seul a un risque d’erreur si on a qu’un seul échantillon & risque de crash + incompatibilité avec MLflow donc on ajoute cpu
            labels = self.label_encoder.inverse_transform(preds)
            return labels.tolist()
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            raise e

# -----------------------------
# LOG MODEL TO MLFLOW
# -----------------------------
if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Transformer_Models")

        print("[DEBUG] Starting MLflow run...")
        with mlflow.start_run(run_name=RUN_NAME) as run:
            print(f"[DEBUG] Active run id: {run.info.run_id}")

            # Example input for signature inference
            example = pd.DataFrame({"text": ["example ticket about billing problem"]})

            # Load test CSV and rename columns if needed
            test_csv_path = os.path.abspath(os.path.join(BASE_DIR, "../../data/processed/sample.csv"))
            if os.path.exists(test_csv_path):
                print(f"[DEBUG] Found test data: {test_csv_path}")
                test_df = pd.read_csv(test_csv_path)
                # Rename CSV columns to match model input
                if "Document" in test_df.columns and "Topic_group" in test_df.columns:
                    test_df = test_df.rename(columns={"Document": "text", "Topic_group": "label"})
                    print("[DEBUG] Renaming columns: Document → text, Topic_group → label")

                    fake_context = PythonModelContext(artifacts={"transformer_model": MODEL_DIR}, model_config={}) #ajouté pour test signature
                    wrapper = TransformerWrapper()
                    wrapper.load_context(fake_context) #aulieu de wrapper.load_context(artifacts_path=MODEL_DIR) car on utilise le contexte simulé, pas un chemin direct

                    preds = wrapper.predict(None, test_df[["text"]])
                    acc = accuracy_score(test_df["label"], preds)
                    f1 = f1_score(test_df["label"], preds, average="weighted")
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    print(f"[DEBUG] Metrics logged → accuracy={acc:.3f}, f1={f1:.3f}")
                else:
                    print("[WARNING] Test CSV missing required columns (Document, Topic_group)")
            else:
                print("[INFO] No test CSV found — skipping metric logging.")

            #ajouté
            fake_context = PythonModelContext(artifacts={"transformer_model": MODEL_DIR}, model_config={})
            sig_wrapper = TransformerWrapper()
            sig_wrapper.load_context(fake_context)
            example_output = sig_wrapper.predict(None, example)
            signature = mlflow.models.infer_signature(example, example_output)

            print("[DEBUG] Logging pyfunc model to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="transformer_model",
                python_model=TransformerWrapper(),
                artifacts={"transformer_model": MODEL_DIR},
                input_example=example,
                signature=signature,
                registered_model_name="CallCenterTransformer"
                
            )

            

            print(f"[SUCCESS] Model logged successfully ✅")
            print(f"View in MLflow UI → http://127.0.0.1:5000/#/experiments")

    except Exception as e:
        print(f"[FATAL ERROR] Something went wrong during MLflow logging: {e}")
        raise e
