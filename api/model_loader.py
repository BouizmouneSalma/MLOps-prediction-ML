from mlflow.tracking import MlflowClient
import pickle
import mlflow
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_best_model(model_name: str = "diabetes-prediction-model", stage: str = "Production") -> Tuple[Any, Any, Dict[str, Any]]:

    try:
        client = MlflowClient()
        
        # production model 
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not model_versions:
            raise Exception(f"No model found in '{stage}' stage for '{model_name}'")

        # model info        
        model_version = model_versions[0]
        run_id = model_version.run_id
        version = model_version.version
        
        logger.info(f"Loading {stage} model: {model_name} v{version} (run: {run_id})")
        
        # Load model
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load scaler from artifacts
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor/scaler.pkl"
        )
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        run = client.get_run(run_id)
        
        # metadata
        metadata = {
            "run_id": run_id,
            "model_type": run.data.params.get("model_type", "Unknown"),
            "val_roc_auc": run.data.metrics.get("val_roc_auc"),
            "model_version": version,
            "stage": stage,
            "model_name": model_name
        }
        
        logger.info(f"Model loaded: {metadata['model_type']} v{version} (ROC-AUC: {metadata['val_roc_auc']:.4f})")
        
        return model, scaler, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise