import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

# ==============================
# 1️⃣  Configuration
# ==============================
DATA_PATH = "data/raw/all_tickets_processed_improved_v3.csv"
MODEL_PATH = "models/tfidf_svm_model.pkl"
MLFLOW_TRACKING_URI = "mlruns"
REGISTERED_MODEL_NAME = "tfidf_svm_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("TFIDF_SVM_Ticket_Classification")

# ==============================
# 2️⃣  Chargement des données
# ==============================
print("📂 Chargement des données...")
df = pd.read_csv(DATA_PATH)

# ⚠️ Vérifie que le CSV contient bien les colonnes nécessaires
if "Document" not in df.columns or "Topic_group" not in df.columns:
    raise ValueError("Le fichier CSV doit contenir les colonnes 'Document' et 'Topic_group'.")

df = df.dropna(subset=["Document", "Topic_group"])
X = df["Document"]
y = df["Topic_group"]

# ==============================
# 3️⃣  Split des données
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4️⃣  Définir le pipeline TF-IDF + SVM calibré
# ==============================
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

svm = LinearSVC(random_state=42)
calibrated_svm = CalibratedClassifierCV(svm, cv=5)  # calibration des probabilités

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words="english", sublinear_tf=True)),
    ("svm", CalibratedClassifierCV(LinearSVC(random_state=42), cv=5))
])

# ==============================
# 5️⃣  Entraînement + suivi MLflow
# ==============================
with mlflow.start_run():
    print("🚀 Entraînement du modèle TF-IDF + SVM...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # ==============================
    # 6️⃣  Évaluation
    # ==============================
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # Rapport complet
    print("\n📊 Classification report :\n")
    print(classification_report(y_test, y_pred))

    # ==============================
    # 7️⃣  Log dans MLflow
    # ==============================
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("model", "LinearSVC (Calibrated)")
    mlflow.log_param("max_features", 8000)
    mlflow.log_param("ngram_range", "(1,2)")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # ==============================
    # 8️⃣  Sauvegarde du modèle
    # ==============================
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)

    # Signature MLflow
    X_sample = X_train.iloc[:5]
    signature = infer_signature(
        X_sample, pipeline.predict(X_sample)
    )

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_sample.to_list(),
        registered_model_name=REGISTERED_MODEL_NAME
    )

    client = mlflow.MlflowClient()
    latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
    if latest_versions:
        latest_version = latest_versions[0].version
        print(f"📦 Dernière version enregistrée du modèle '{REGISTERED_MODEL_NAME}': {latest_version}")
    else:
        print(f"⚠️ Aucun modèle trouvé pour '{REGISTERED_MODEL_NAME}'")
    
    print(f"💾 Modèle sauvegardé dans {MODEL_PATH}")

print("🎉 Entraînement terminé et suivi avec MLflow !")


"""
import os
import mlflow
import pandas as pd
import re
import string
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def setup_mlflow_tracking():
    # Create directories
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # Use environment variables to separate tracking and artifacts
    os.environ["MLFLOW_TRACKING_URI"] = "file:///" + os.path.abspath("mlruns").replace("\\", "/")
    os.environ["MLFLOW_ARTIFACT_URI"] = "file:///" + os.path.abspath("artifacts").replace("\\", "/")
    
    # Alternative: Use relative paths (works better in Docker)
    # os.environ["MLFLOW_TRACKING_URI"] = "file:///mlruns"
    # os.environ["MLFLOW_ARTIFACT_URI"] = "file:///artifacts"
    
    logger.info(f"Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    logger.info(f"Artifact URI: {os.environ.get('MLFLOW_ARTIFACT_URI')}")

def clean_text(text: str) -> str:
    try:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return " ".join(words)
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def load_dataset(csv_path: str) -> pd.DataFrame:
    
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")
        
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Le dataset est vide.")
        
        if "Document" not in df.columns or "Topic_group" not in df.columns:
            raise ValueError("Le dataset doit contenir les colonnes 'Document' et 'Topic_group'.")
        
        df["Cleaned_Document"] = df["Document"].apply(clean_text)
        logger.info(f"Loaded and cleaned dataset from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def load_params(file_path: str) -> dict:
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
            if params is None or "tfidf_svm" not in params:
                raise ValueError("Le fichier params.yaml est vide ou n'a pas de section 'tfidf_svm'.")
            tfidf_params = params["tfidf_svm"]
            tfidf_params.setdefault("max_features", 5000)
            tfidf_params.setdefault("test_size", 0.2)
            tfidf_params.setdefault("random_state", 42)
            return tfidf_params
    except Exception as e:
        logger.error(f"Error loading params: {e}")
        raise

def train_tfidf_svm(df: pd.DataFrame, params: dict):
    
    try:
        setup_mlflow_tracking()
        
        X = df["Cleaned_Document"]
        y = df["Topic_group"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
        )
        logger.info("Completed train-test split")

        experiment_name = "TFIDF_SVM"
        try:
            mlflow.set_experiment(experiment_name)
        except:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        logger.info(f"Using experiment '{experiment_name}'")

        with mlflow.start_run(run_name="tfidf_svm_run") as run:
            mlflow.log_params(params)

            # Create and train pipeline
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=params["max_features"], ngram_range=(1, 2))),
                ("svm", CalibratedClassifierCV(LinearSVC(random_state=params["random_state"])))
            ])
            pipeline.fit(X_train, y_train)
            logger.info("Pipeline trained")

            # Evaluate
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info(f"Evaluation - Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            print(f"mlflow log metrics, F1 Score: {f1:.4f}")
            mlflow.log_metric("f1_score", f1)
            print("mlflow log metrics, Classification Report saved to classification_report.json")
            mlflow.log_dict(report, "classification_report.json")
            
            logger.info(f"Metrics logged - Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Log model with error handling
            try:
                input_example = X_train[:1].values.reshape(1, -1)
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="tfidf_svm_model",
                    input_example=input_example,
                    conda_env={
                        'channels': ['conda-forge'],
                        'dependencies': [
                            'python=3.9',
                            'scikit-learn',
                            'pandas',
                            'numpy',
                            'nltk'
                        ]
                    }
                )
                logger.info("Model successfully logged to MLflow")
            except Exception as model_error:
                logger.warning(f"Failed to log model to MLflow: {model_error}")
                logger.info("Model will be saved locally instead")

            # Try model registration with fallback
            try:
                client = mlflow.MlflowClient()
                model_uri = f"runs:/{run.info.run_id}/tfidf_svm_model"
                registered_model = mlflow.register_model(model_uri, "tfidf_svm_model")
                client.set_model_version_tag(
                    name="tfidf_svm_model",
                    version=registered_model.version,
                    key="stage",
                    value="Production"
                )
                logger.info(f"Model registered and promoted to Production (version {registered_model.version})")
            except Exception as reg_error:
                logger.warning(f"Model registration failed: {reg_error}")

            # Always save model locally as backup
            save_model(pipeline, output_dir="models")
            
        return pipeline
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

def save_model(pipeline, output_dir="models"):
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "tf_idf_svm_model.pkl")
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved locally: {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting TF-IDF SVM training pipeline")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        params = load_params("params.yaml")
        csv_path = "data/raw/all_tickets_processed_improved_v3.csv"
        df = load_dataset(csv_path)
        pipeline = train_tfidf_svm(df, params)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution : {e}")
        raise
"""