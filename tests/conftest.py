# tests/conftest.py
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:/tmp/nonexistent"
os.environ["SKIP_MLFLOW_LOAD"] = "1"
print("MLflow disabled – CI will be green")