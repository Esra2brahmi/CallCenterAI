CallCenterAI – Intelligent Customer Ticket Classification System






📖 Description

CallCenterAI est une solution MLOps complète pour la classification automatique des tickets clients (emails, chat, téléphone) en différentes catégories métiers (ex. Facturation, Problème technique, Accès compte, etc.).

Le projet intègre :

Deux approches NLP :

TF-IDF + SVM (scikit-learn)

Transformer (Hugging Face – multilingue FR/EN/AR)

Microservices FastAPI pour chaque modèle et un agent IA pour orchestrer les prédictions.

Conteneurisation avec Docker et orchestration via Docker Compose.

Pipeline MLOps complet avec MLflow, DVC, CI/CD GitHub Actions.

Monitoring et observabilité avec Prometheus et Grafana.

🏗 Architecture
graph TB
    User[Client/Ticket Submission] --> Agent[AI Agent Service]
    Agent --> TFIDF[TF-IDF + SVM Service]
    Agent --> Transformer[Transformer Service]
    TFIDF --> MLflow_TFIDF[MLflow Tracking]
    Transformer --> MLflow_Transformer[MLflow Tracking]
    MLflow_TFIDF --> DVC[DVC Pipeline]
    MLflow_Transformer --> DVC
    Prometheus --> Grafana[Dashboard Grafana]


Structure du dépôt :

CallCenterAI/
CallCenterAI/
├─ .github/
│  └─ workflows/
│     ├─ lint-test.yml
│     ├─ docker-build.yml
│     ├─ ci-agent.yml
│     ├─ ci-transformer.yml
│     └─ ci-tfidf.yml
├─ docker/
│  ├─ tfidf_svc.Dockerfile
│  ├─ transformersService.Dockerfile
│  ├─ agent_service.Dockerfile
│  └─ docker-compose.override.yml
├─ src/
|  └─ models/
│  |   ├─ mlflow_tfidf.py              # MLflow model loader for TF-IDF + SVM
│  |   └─ mlflow_transformer.py        # MLflow model loader for transformers
│  |    
│  |   
│  └─ services/
│     ├─ tfidf_service/
│     │  ├─ __init__.py
│     │  ├─ app.py                   # FastAPI app + endpoints
│     │  
│     │  
│     │  
│     ├─ transformer_service/
│     │  ├─ __init__.py
│     │  ├─ serviceFromMLFlow.py     # FastAPI app + transformer inference
│     │  
│     │  
│     └─ agent_Ai/
│        ├─ __init__.py
│        ├─ appGPT.py                 # fast api  service
│        ├─ generate_router_training.py                   
│        └─ train_router.py                  # to train model  
│        
├─ scripts/
│  ├─ train_tfidf.py                # training pipeline for TF-IDF + SVM
│  ├─ train_transformer.py          # fine-tune/pack transformer model
│  └─ serve_local_mlflow.sh
├─ requirements/
│  ├─ base.txt
│  ├─ transformer.txt
│  └─ dev.txt
├─ tests/
│  ├─ unit/
│  │  ├─ test_tfidf_preprocessing.py
│  │  ├─ test_transformer_loader.py
│  │  └─ test_agent_logic.py
│  └─ integration/
│     └─ test_integration.py        # integration tests using TestClient
├─ notebooks/
│  ├─ eda.ipynb
│  └─ model_experiments.ipynb
├─ models/                          # local exported model artifacts
│  ├─ tfidf/
│  └─ transformer/
├─ mlruns/                          # MLflow runs / registry (local)
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ README.md
├─ dvc.yaml
├─ params.yaml
├─ .dvc/
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
├─ .dockerignore
├─ Makefile
├─ README.md
└─ architecture                      

⚡ Fonctionnalités

Agent IA intelligent :

Sélection du modèle approprié (TF-IDF ou Transformer)

Nettoyage des données sensibles (Scrub PII)

Retourne la prédiction et la confiance avec explication

Expose des métriques Prometheus

TF-IDF + SVM :

Pipeline Scikit-learn

Probabilités calibrées

Logging métriques dans MLflow

Transformer Multilingue :

Fine-tuning avec Hugging Face

Prise en charge du français, anglais et arabe

MLOps :

DVC pour pipeline data/model

MLflow pour suivi des runs et registry

CI/CD via GitHub Actions (tests, lint, build, push Docker images)

Monitoring :

Dashboard Grafana (latence, requêtes, erreurs)

Prometheus scraping endpoints /metrics

🚀 Installation et Lancement

Cloner le dépôt :

git clone https://github.com/maryem38/CallCenterAI.git
cd CallCenterAI


Configurer l’environnement :

cp .env.example .env
--> pip install requirements.txt/

Construire les images Docker : docker-compose build


Lancer les services :

docker-compose up -d


Accéder aux APIs :

TF-IDF Service : http://localhost:8001/predict

Transformer Service : http://localhost:8002/predict

Agent IA : http://localhost:8000/predict

Accéder au monitoring :

Prometheus : http://localhost:9090

Grafana : http://localhost:3000




CI/CD GitHub Actions gère le linting (black, flake8, isort) et le scan sécurité (Trivy, Bandit).

📊 Dataset

Source : IT Service Ticket Classification – Kaggle

Taille : ~47 000 tickets

Colonnes : Document (texte du ticket), Topic_group (catégorie)

Catégories : Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, etc.

🛠 Stack Technologique

Langage : Python 3.11

API Framework : FastAPI

ML/NLP : scikit-learn (TF-IDF + SVM), Hugging Face Transformers

MLOps : MLflow, DVC, Docker, Docker Compose, GitHub Actions

Monitoring : Prometheus, Grafana

Tests & Qualité : pytest

📈 Pipeline MLOps

Préparation des données (dvc.yaml)

Entraînement TF-IDF + SVM (src/models/mlflow_tfidf.py)

Fine-tuning Transformer (src/models/mlflow_transformer.py)

Déploiement via Docker & Docker Compose  (docker-compose up --build)

Suivi des métriques et modèles dans MLflow (mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts --disable-security-middleware)

Monitoring et alertes via Prometheus/Grafana

📄 Références

FastAPI Documentation

scikit-learn Documentation

Hugging Face Transformers

MLflow

DVC

Prometheus

Grafana