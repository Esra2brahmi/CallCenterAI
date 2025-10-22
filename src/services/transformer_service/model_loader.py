import mlflow
import torch
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility class to load transformer model from MLflow"""
    
    def __init__(self, mlflow_uri: str = None, model_path: str = None):
        """
        Initialize model loader
        
        Args:
            mlflow_uri: MLflow tracking URI
            model_path: Local path to model (fallback)
        """
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.model_path = model_path or "../../src/models/mlflow_logged_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_from_registry(self, model_name: str, stage: str = "Production"):
        """
        Load model from MLflow Model Registry
        
        Args:
            model_name: Name of registered model
            stage: Model stage (Production, Staging, etc.)
            
        Returns:
            tuple: (model, tokenizer, label_encoder)
        """
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            model_uri = f"models:/{model_name}/{stage}"
            
            logger.info(f"Loading model from registry: {model_uri}")
            logged_model = mlflow.pyfunc.load_model(model_uri)
            
            return self._extract_components(logged_model)
            
        except Exception as e:
            logger.error(f"Failed to load from registry: {e}")
            raise
    
    def load_from_run(self, run_id: str):
        """
        Load model from specific MLflow run
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            tuple: (model, tokenizer, label_encoder)
        """
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            model_uri = f"runs:/{run_id}/model"
            
            logger.info(f"Loading model from run: {model_uri}")
            logged_model = mlflow.pyfunc.load_model(model_uri)
            
            return self._extract_components(logged_model)
            
        except Exception as e:
            logger.error(f"Failed to load from run: {e}")
            raise
    
    def load_from_local(self, path: str = None):
        """
        Load model from local filesystem
        
        Args:
            path: Path to model directory
            
        Returns:
            tuple: (model, tokenizer, label_encoder)
        """
        try:
            model_path = path or self.model_path
            logger.info(f"Loading model from local path: {model_path}")
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            logged_model = mlflow.pyfunc.load_model(model_path)
            
            return self._extract_components(logged_model)
            
        except Exception as e:
            logger.error(f"Failed to load from local: {e}")
            raise
    
    def load_model(self, 
                   model_name: str = None, 
                   run_id: str = None, 
                   local_path: str = None,
                   stage: str = "Production"):
        """
        Load model with automatic fallback strategy
        
        Priority:
        1. Model Registry (if model_name provided)
        2. Specific Run (if run_id provided)
        3. Local path (if local_path provided or default)
        
        Args:
            model_name: Name in Model Registry
            run_id: MLflow run ID
            local_path: Local filesystem path
            stage: Model stage for registry
            
        Returns:
            tuple: (model, tokenizer, label_encoder)
        """
        # Try model registry first
        if model_name:
            try:
                return self.load_from_registry(model_name, stage)
            except Exception as e:
                logger.warning(f"Registry load failed: {e}")

        # Then try specific run
        if run_id:
            try:
                return self.load_from_run(run_id)
            except Exception as e:
                logger.warning(f"Run load failed: {e}")

        # Finally try local path
        try:
            return self.load_from_local(local_path)
        except Exception as e:
            logger.error(f"All loading attempts failed: {e}")
            raise