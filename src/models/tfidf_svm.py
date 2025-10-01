# src/models/tfidf_svm.py
"""purpose of using tf-idf:
    mesure statistique utilisée pour évaluer l'importance d'un mot dans un document par rapport à un ensemble de documents
"""
"""
TF-IDF: Turns text into numbers representing word importance.
SVM: Uses those numbers to classify text into categories.
"""

import os
import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from src.utils.data_utils import load_data, save_sample  # use your helper functions
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -------------------------
# 1️⃣ Text preprocessing
# -------------------------
def clean_text(text: str) -> str:
    """Clean text: lowercase, remove punctuation, numbers, stopwords."""
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
    df = pd.read_csv(csv_path)
    df["Cleaned_Document"] = df["Document"].apply(clean_text)
    return df

# -------------------------
# 3️⃣ Train TF-IDF + SVM
# -------------------------
#DataFrame (dans pandas) est une structure de données en 2 dimensions (tableau),
def train_tfidf_svm(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df["Cleaned_Document"]
    y = df["Topic_group"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Linear SVM classifier
    svm = LinearSVC(random_state=random_state)
    svm.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = svm.predict(X_test_tfidf)
    print("Classification report:\n", classification_report(y_test, y_pred))

    
    #we can use thoose instructions to make the code more fluide, maintainable, safer and it allow probability calibration
    """
    # Pipeline: TF-IDF + Calibrated SVM
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=params["max_features"], ngram_range=(1, 2))),
        ("svm", CalibratedClassifierCV(LinearSVC(random_state=params["random_state"])))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Log to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # Adjust if needed
    mlflow.set_experiment("TFIDF_SVM")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(pipeline, "tfidf_svm_model")
    """

    return tfidf, svm

# -------------------------
# 4️⃣ Save models
# -------------------------
def save_model(tfidf, svm, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(svm, os.path.join(output_dir, "svm_model.pkl"))
    print(f"Models saved in {output_dir}/")
