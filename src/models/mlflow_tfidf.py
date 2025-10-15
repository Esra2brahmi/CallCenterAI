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

def clean_text(text: str) -> str:
    """Clean text: lowercase, remove punctuation, numbers, stopwords, and apply lemmatization."""
    try:
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return " ".join(words)
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV from data/raw and apply cleaning."""
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
    """Load parameters from YAML file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        with open(file_path, "r") as f:
            content = f.read()
            logger.info("Raw YAML content:\n%s", content)
            params = yaml.safe_load(content)
            if params is None or "tfidf_svm" not in params:
                raise ValueError("Le fichier params.yaml est vide ou n'a pas de section 'tfidf_svm'.")
            tfidf_params = params["tfidf_svm"]
            logger.info("tfidf_svm params: %s", tfidf_params)
            if not isinstance(tfidf_params, dict):
                raise TypeError(f"Expected 'tfidf_svm' to be a dictionary, got {type(tfidf_params)}: {tfidf_params}")
            # Set default values if missing
            tfidf_params.setdefault("max_features", 5000)
            tfidf_params.setdefault("test_size", 0.2)
            tfidf_params.setdefault("random_state", 42)
            return tfidf_params
    except Exception as e:
        logger.error(f"Error loading params: {e}")
        raise

def train_tfidf_svm(df: pd.DataFrame, params: dict):
    """Train a TF-IDF + SVM pipeline and log to MLflow."""
    try:
        if not isinstance(params, dict):
            raise TypeError(f"Expected params to be a dictionary, got {type(params)}: {params}")
        
        X = df["Cleaned_Document"]
        y = df["Topic_group"]

        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
        )
        logger.info("Completed train-test split")

        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri("http://localhost:5000")
        experiment_name = "TFIDF_SVM"
        experiments = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")

        if experiments:
            experiment_id = experiments[0].experiment_id
            logger.info(f"Found existing experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")

        mlflow.set_experiment("TFIDF_SVM")

        with mlflow.start_run(run_name="tfidf_svm_parent_run") as parent_run:
            mlflow.log_param("parent_run_type", "TF-IDF_SVM_training")
            mlflow.log_params(params)
            logger.info("Started parent MLflow run: %s", parent_run.info.run_id)

            # Experiment with multiple max_features values
            max_features_values = [5000]  # Expanded for hyperparameter tuning
            for max_features in max_features_values:
                with mlflow.start_run(run_name=f"child_run_max_features_{max_features}", nested=True) as child_run:
                    mlflow.log_param("max_features", max_features)
                    mlflow.log_params(params)
                    logger.info(f"Started child run for max_features={max_features}: {child_run.info.run_id}")

                    # Define and train pipeline
                    pipeline = Pipeline([
                        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
                        ("svm", CalibratedClassifierCV(LinearSVC(random_state=params["random_state"])))
                    ])
                    pipeline.fit(X_train, y_train)
                    logger.info(f"Trained pipeline with max_features={max_features}")

                    # Evaluate model
                    y_pred = pipeline.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    report = classification_report(y_test, y_pred, output_dict=True)

                    # Log metrics and report
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_dict(report, f"classification_report_{max_features}.json")
                    logger.info(f"Results for max_features={max_features}:")
                    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

                    # Log model with input example
                    input_example = X_train[:1].to_frame().to_dict()
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path=f"tfidf_svm_model_{max_features}",
                        registered_model_name="tfidf_svm_model",
                        input_example=input_example
                    )
                    logger.info(f"Model logged as: tfidf_svm_model_{max_features}")

                    # Register model and set stage using tags
                    client = mlflow.MlflowClient()
                    model_uri = f"runs:/{child_run.info.run_id}/tfidf_svm_model_{max_features}"
                    registered_model = mlflow.register_model(model_uri, "tfidf_svm_model")
                    logger.info(f"Registered model 'tfidf_svm_model' version {registered_model.version}")

                    # Set model stage using tags (to replace deprecated transition_model_version_stage)
                    client.set_model_version_tag(
                        name="tfidf_svm_model",
                        version=registered_model.version,
                        key="stage",
                        value="Production"
                    )
                    logger.info(f"Model promoted to Production (version {registered_model.version})")

        return pipeline
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

def save_model(pipeline, output_dir="models"):
    """Save the trained model to disk."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(output_dir, "tf_idf_svm_model.pkl"))
        logger.info(f"Model saved in {output_dir}/tf_idf_svm_model.pkl")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Current working directory: %s", os.getcwd())
        params = load_params("params.yaml")
        csv_path = "data/raw/all_tickets_processed_improved_v3.csv"
        df = load_dataset(csv_path)
        pipeline = train_tfidf_svm(df, params)
        # save_model(pipeline)  # Uncomment if you want to save the model locally
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution : {e}")