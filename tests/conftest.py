# tests/conftest.py
import os
import pytest

# Force MLflow to use the local dummy registry BEFORE any import
os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")