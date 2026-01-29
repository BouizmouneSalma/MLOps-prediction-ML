from mlflow.tracking import MlflowClient
import pickle
import mlflow
from typing import Tuple, Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI to parent directory
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", 
    f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))}"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

def load_best_model(model_name: str = "diabetes-prediction-model", stage: str = "Production", fallback_to_best_run: bool = True) -> Tuple[Any, Any, Dict[str, Any]]:

    try:
        client = MlflowClient()
        
        # production model 
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not model_versions:
            if not fallback_to_best_run:
                raise Exception(f"No model found in '{stage}' stage for '{model_name}'")
            
            logger.warning(f"No model in {stage} stage. Falling back to best run from experiment.")
            return _load_best_from_experiment()

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
        if fallback_to_best_run:
            logger.warning(f"Failed to load from registry: {str(e)}. Trying best run from experiment.")
            return _load_best_from_experiment()
        logger.error(f"Error loading model: {str(e)}")
        raise


def _load_best_from_experiment(experiment_name: str = "diabetes_model_comparaison") -> Tuple[Any, Any, Dict[str, Any]]:
    """Fallback: Load best model from experiment runs"""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            raise Exception(f"Experiment '{experiment_name}' not found")
        
        # Get best model based on validation ROC-AUC
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_roc_auc DESC"],
            max_results=1
        )
        
        if not runs:
            raise Exception("No runs found in experiment")
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        
        logger.info(f"Loading best model from experiment run: {run_id}")
        
        # Load model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load scaler
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor/scaler.pkl"
        )
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        # Prepare metadata
        metadata = {
            "run_id": run_id,
            "model_type": best_run.data.params.get("model_type", "Unknown"),
            "val_roc_auc": best_run.data.metrics.get("val_roc_auc"),
            "model_version": best_run.data.tags.get("mlflow.runName", run_id[:8]),
            "stage": "Not Registered",
            "model_name": "N/A"
        }
        
        logger.info(f"Model loaded from experiment: {metadata['model_type']} (ROC-AUC: {metadata['val_roc_auc']:.4f})")
        
        return model, scaler, metadata
        
    except Exception as e:
        logger.error(f"Error loading model from experiment: {str(e)}")
        raise