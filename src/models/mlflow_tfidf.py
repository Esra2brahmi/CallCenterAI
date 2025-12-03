import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from mlflow.models.signature import infer_signature

# ==============================
# 1. Configuration
# ==============================
DATA_PATH = "data/raw/all_tickets_processed_improved_v3.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "tfidf_svm_ticket_classifier"  # Better name (no spaces, clear intent)

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("TFIDF_SVM_Ticket_Classification")

# Optional: define conda environment (recommended for production readiness)
conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "python=3.10.*",
        "pip",
        {
            "pip": [
                "mlflow",
                "scikit-learn",
                "pandas",
                "cloudpickle",
            ],
        },
    ],
    "name": "mlflow-env",
}

# ==============================
# 2. Chargement des données
# ==============================
print("Chargement des données...")
df = pd.read_csv(DATA_PATH)

required_columns = ["Document", "Topic_group"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Le CSV doit contenir les colonnes : {required_columns}")

df = df.dropna(subset=required_columns)
X = df["Document"]
y = df["Topic_group"]

# ==============================
# 3. Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. Pipeline
# ==============================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )),
    ("svm", CalibratedClassifierCV(
        LinearSVC(random_state=42, class_weight='balanced'),  # added class_weight for better handling of imbalance
        cv=5,
        method='sigmoid'
    ))
])

# ==============================
# 5. Entraînement + MLflow Tracking
# ==============================
with mlflow.start_run(run_name="tfidf_svm_calibrated_v1") as run:
    print("Entraînement du modèle...")
    pipeline.fit(X_train, y_train)

    # Prédictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if len(pipeline.classes_) == 2 else None

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ==============================
    # 6. Log params & metrics
    # ==============================
    mlflow.log_params({
        "vectorizer": "TfidfVectorizer",
        "max_features": 8000,
        "ngram_range": "(1,2)",
        "stop_words": "english",
        "sublinear_tf": True,
        "classifier": "CalibratedClassifierCV(LinearSVC)",
        "calibration_method": "sigmoid",
        "cv_folds": 5,
        "test_size": 0.2,
        "random_state": 42
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "f1_weighted": f1
    })

    # ==============================
    # 7. Signature & input example
    # ==============================
    sample_input = X_train.iloc[:10].tolist()
    predictions_sample = pipeline.predict(sample_input)
    signature = infer_signature(sample_input, predictions_sample)

    # ==============================
    # 8. Enregistrement du modèle dans MLflow (SANS sauvegarde locale !)
    # ==============================
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=sample_input,
        conda_env=conda_env,
        registered_model_name=REGISTERED_MODEL_NAME,
        metadata={"description": "TF-IDF + Calibrated LinearSVC for ticket classification"}
    )

    # Optionnel : ajouter des tags
    mlflow.set_tags({
        "model_type": "text_classification",
        "framework": "sklearn",
        "team": "nlp-support",
        "stage": "staging"
    })

    # Récupérer la version enregistrée
    client = mlflow.MlflowClient()
    latest_version_info = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
    if latest_version_info:
        version = latest_version_info[0].version
        print(f"Modèle enregistré avec succès !")
        print(f"   → Nom : {REGISTERED_MODEL_NAME}")
        print(f"   → Version : {version}")
        print(f"   → Run ID  : {run.info.run_id}")
    else:
        print("Modèle loggué mais pas encore visible dans le registry (normal lors du premier enregistrement)")

print("Entraînement et enregistrement MLflow terminés avec succès !")








"""import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

# ==============================
# 1️⃣ Configuration
# ==============================
DATA_PATH = "data/raw/all_tickets_processed_improved_v3.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "TFIDF_SVM_Ticket_Classification"
REGISTERED_MODEL_NAME = "tfidf_svm_model"

mlflow.set_experiment(EXPERIMENT_NAME)

# ==============================
# 2️⃣ Load Data
# ==============================
print("📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Document", "Topic_group"])
X = df["Document"]
y = df["Topic_group"]

# ==============================
# 3️⃣ Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4️⃣ Define Pipeline
# ==============================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )),
    ("svm", CalibratedClassifierCV(LinearSVC(random_state=42), cv=5))
])

# ==============================
# 5️⃣ Train & Log with MLflow
# ==============================
with mlflow.start_run() as run:
    print("🚀 Training TF-IDF + SVM model...")
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # ==============================
    # 6️⃣ Evaluation
    # ==============================
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("\n📊 Classification report :\n")
    print(classification_report(y_test, y_pred))

    # ==============================
    # 7️⃣ Log Parameters & Metrics
    # ==============================
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("model", "LinearSVC (Calibrated)")
    mlflow.log_param("max_features", 8000)
    mlflow.log_param("ngram_range", "(1,2)")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # ==============================
    # 8️⃣ Log the Model
    # ==============================
    # Signature for model input/output
    X_sample = X_train.iloc[:5]
    signature = infer_signature(X_sample, pipeline.predict(X_sample))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_sample.to_list(),
        registered_model_name=REGISTERED_MODEL_NAME
    )

    # Transition registered model to Production
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"💾 Model logged and registered as '{REGISTERED_MODEL_NAME}' in Production")
    print("Artifact URI:", mlflow.get_artifact_uri())

print("🎉 Training finished and tracked with MLflow!")
"""