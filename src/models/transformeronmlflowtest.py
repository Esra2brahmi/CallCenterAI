# src/models/mlflow_transformer.py
import os
import sys
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PythonModelContext
import torch
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

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

print("[DEBUG] All model files found")

# -----------------------------
# DEFINE CUSTOM PYFUNC MODEL
# -----------------------------
class TransformerWrapper(mlflow.pyfunc.PythonModel):
    # PythonModelContext provides access to artifacts and config: est un objet fourni par MLflow lors du chargement d’un modèle PythonModel.
    #Il contient tout ce dont ton modèle a besoin pour se charger : chemin reele de l'artifact, configuration, etc.

    def load_context(self, context: PythonModelContext):
        try:
            print("[DEBUG] Loading model artifacts via MLflow context...")
            model_path = context.artifacts["transformer_model"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
            self.model.eval()
            print("[DEBUG] Model, tokenizer, and label encoder loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed in load_context(): {e}")
            raise e

    def predict(self, context, model_input: pd.DataFrame):
        try:
            texts = model_input["text"].tolist()
            print(f"[DEBUG] Predicting on {len(texts)} texts")
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=128
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
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

            # Example input for signature
            example_input = pd.DataFrame({"text": ["example ticket about billing problem"]})

            # --- TEST METRICS (optional) ---
            test_csv_path = os.path.abspath(os.path.join(BASE_DIR, "../../data/processed/sample.csv"))
            if os.path.exists(test_csv_path):
                print(f"[DEBUG] Found test data: {test_csv_path}")
                test_df = pd.read_csv(test_csv_path)

                if "Document" in test_df.columns and "Topic_group" in test_df.columns:
                    test_df = test_df.rename(columns={"Document": "text", "Topic_group": "label"})
                    print("[DEBUG] Columns renamed: Document → text, Topic_group → label")

                    # Simulate MLflow context for local testing
                    fake_context = PythonModelContext(artifacts={"transformer_model": MODEL_DIR}, model_config={})
                    wrapper = TransformerWrapper()
                    wrapper.load_context(fake_context)

                    preds = wrapper.predict(None, test_df[["text"]])
                    acc = accuracy_score(test_df["label"], preds)
                    f1 = f1_score(test_df["label"], preds, average="weighted")
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    print(f"[DEBUG] Metrics logged → accuracy={acc:.4f}, f1={f1:.4f}")
                else:
                    print("[WARNING] Test CSV missing required columns: Document, Topic_group")
            else:
                print("[INFO] No test CSV found — skipping metrics.")

            # --- INFER SIGNATURE WITH REAL PREDICTION ---
            fake_context = PythonModelContext(artifacts={"transformer_model": MODEL_DIR}, model_config={})
            sig_wrapper = TransformerWrapper()
            sig_wrapper.load_context(fake_context)
            example_output = sig_wrapper.predict(None, example_input)
            
            #description formelle de l’entrée et de la sortie de ton modèle. (mlflow fait automatiquement pour toi, ici on le fait manuellement pour s’assurer que c’est correct)
            signature = mlflow.models.infer_signature(example_input, example_output)

            # --- LOG MODEL ---
            print("[DEBUG] Logging model to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="transformer_model",
                python_model=TransformerWrapper(),
                artifacts={"transformer_model": MODEL_DIR},
                input_example=example_input,
                signature=signature,
                registered_model_name="CallCenterTransformer"
            )

            print(f"[SUCCESS] Model logged and registered as 'CallCenterTransformer'")
            print(f"View run: http://127.0.0.1:5000/#/experiments/145277697902719720/runs/{run.info.run_id}")

    except Exception as e:
        print(f"[FATAL ERROR] MLflow logging failed: {e}")
        raise e