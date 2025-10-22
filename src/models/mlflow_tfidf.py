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
