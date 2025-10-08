# src/models/tfidf_svm.py
"""purpose of using tf-idf:
    mesure statistique utilisée pour évaluer l'importance d'un mot dans un document par rapport à un ensemble de documents
"""
"""
TF-IDF: Turns text into numbers representing word importance.
SVM: Uses those numbers to classify text into categories.
"""

import os
from fastapi import params
import mlflow
import pandas as pd
import re
import string
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import yaml
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
# Download stopwords if not already
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# -------------------------
# 1️⃣ Text preprocessing
# -------------------------
def clean_text(text: str) -> str:
    """Clean text: lowercase, remove punctuation, numbers, stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------------
# 2️⃣ Load dataset
# -------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV from data/raw and apply cleaning."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")
    df = pd.read_csv(csv_path)
    if "Document" not in df.columns or "Topic_group" not in df.columns:
        raise ValueError("Le dataset doit contenir les colonnes 'Document' et 'Topic_group'.")
    df["Cleaned_Document"] = df["Document"].apply(clean_text)
    return df

# -------------------------
# 3️⃣ Load parameters
# -------------------------
def load_params(file_path: str) -> dict:
    """Load parameters from YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
        if params is None or "tfidf_svm" not in params:
            raise ValueError("Le fichier params.yaml est vide ou n'a pas de section 'tfidf_svm'.")
        return params["tfidf_svm"]

# -------------------------
# 4️⃣ Train TF-IDF + SVM
# -------------------------
def train_tfidf_svm(df: pd.DataFrame, params: dict):
    """Train a TF-IDF + SVM pipeline and log to MLflow."""
    X = df["Cleaned_Document"]
    y = df["Topic_group"]

    # Split into train/test
    #mlflow.autolog()

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
    )

    mlflow.set_tracking_uri("http://localhost:5000")

    # 🔄 Check for existing experiment using search_experiments
    experiment_name = "TFIDF_SVM"
    experiments = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")

    if experiments:
        # Experiment exists, use its ID
        experiment_id = experiments[0].experiment_id
        print(f"Found existing experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        # Create new experiment if it doesn't exist
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")

    mlflow.set_experiment("TFIDF_SVM")

    with mlflow.start_run(run_name="tfidf_svm_parent_run") as parent_run:
        """mlflow.sklearn.autolog()  # Auto-log parameters, metrics, and models"""
        mlflow.log_params("parent_run_type", "TF-IDF_SVM_training")

        max_features_values = [5000, 10000, 15000]
        for max_features in max_features_values:
            with mlflow.start_run(run_name=f"child_run_max_features_{max_features}", nested=True):
                # Log paramètres spécifiques du child run
                mlflow.log_param("max_features", max_features)
                mlflow.log_params(params)

                # Pipeline: TF-IDF + Calibrated SVM
                pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
                    ("svm", CalibratedClassifierCV(LinearSVC(random_state=params["random_state"])))
                ])

                # Train model
                pipeline.fit(X_train, y_train)

                # Evaluate
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                report = classification_report(y_test, y_pred, output_dict=True)

                # Log metrics and artifacts
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_dict(report, f"classification_report_{max_features}.json")
                mlflow.sklearn.log_model(pipeline, f"tfidf_svm_model_{max_features}")

                print(f"Results for max_features={max_features}:")
                print("Classification report:\n", classification_report(y_test, y_pred))

    return pipeline

# -------------------------
# 5️⃣ Save models
# -------------------------
def save_model(pipeline, output_dir="models"):
    """Save the trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_dir, "tf_idf_svm_model.pkl"))
    print(f"Model saved in {output_dir}/")

# -------------------------
# 6️⃣ Main execution
# -------------------------
if __name__ == "__main__":
    try:
        # Load parameters
        params = load_params("params.yaml")
        
        # Load dataset
        csv_path = "data/processed/sample.csv"
        df = load_dataset(csv_path)
        
        # Train and save model
        pipeline = train_tfidf_svm(df, params)
        save_model(pipeline)
        
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")