import mlflow 
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import pickle
import os
import sys
from datetime import datetime

from models import get_models_and_grids
from data_validation import validate_dataset


df = pd.read_csv("data/data.csv")

# validate dataset
is_valid, validation_report = validate_dataset(df, target_column="Cluster")

if not is_valid:
    print("\nWARNING: Data validation failed. Review issues above.")
    sys.exit(1)

X = df.drop("Cluster", axis=1)
y = df["Cluster"]

RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Fit scaler on training data for production use 
scaler = StandardScaler()
scaler.fit(X_train)  


# dataset info 
dataset_info = {
    "total_samples": len(df),
    "n_features": X.shape[1],
    "feature_names": X.columns.tolist(),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "train_class_distribution": y_train.value_counts().to_dict(),
    "test_class_distribution": y_test.value_counts().to_dict(),
    "target_column": "Cluster",
    "timestamp": datetime.now().isoformat()
}


mlflow.set_experiment("diabetes_model_comparaison")

best_model = None
best_overall_score = 0
best_overall_name = None

for name, (model, param_grid) in get_models_and_grids().items():
    print(f"\n{'='*60}")
    print(f"Running GridSearch for {name}...")
    print(f"{'='*60}")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    current_best_model = grid.best_estimator_
    val_preds_proba = current_best_model.predict_proba(X_test)[:,1]
    val_preds_class = current_best_model.predict(X_test)
    
    # metrics
    val_roc = roc_auc_score(y_test, val_preds_proba)
    val_accuracy = accuracy_score(y_test, val_preds_class)
    val_f1 = f1_score(y_test, val_preds_class, average='weighted')
    val_precision = precision_score(y_test, val_preds_class, average='weighted')
    val_recall = recall_score(y_test, val_preds_class, average='weighted')
    
    cm = confusion_matrix(y_test, val_preds_class)
    
    class_report = classification_report(y_test, val_preds_class, output_dict=True)

    # log dataset and hyperparametres as metrices  
    with mlflow.start_run(run_name=name):

        mlflow.log_metric("dataset_total_samples", dataset_info['total_samples'])
        mlflow.log_metric("dataset_n_features", dataset_info['n_features'])
        mlflow.log_metric("dataset_train_size", dataset_info['train_size'])
        mlflow.log_metric("dataset_test_size", dataset_info['test_size'])
        
        for class_label, count in dataset_info['train_class_distribution'].items():
            mlflow.log_metric(f"train_class_{class_label}_count", count)
        for class_label, count in dataset_info['test_class_distribution'].items():
            mlflow.log_metric(f"test_class_{class_label}_count", count)
        
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("model_type", name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("target_column", dataset_info['target_column'])
        mlflow.log_param("feature_names", ",".join(dataset_info['feature_names']))
        mlflow.log_param("timestamp", dataset_info['timestamp'])
        mlflow.log_param("preprocessing", "StandardScaler")
        
        mlflow.log_metric("cv_best_roc_auc", grid.best_score_)
        mlflow.log_metric("cv_std_roc_auc", grid.cv_results_['std_test_score'][grid.best_index_])
        
        mlflow.log_metric("val_roc_auc", val_roc)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_weighted", val_f1)
        mlflow.log_metric("val_precision_weighted", val_precision)
        mlflow.log_metric("val_recall_weighted", val_recall)
        
        for class_label, metrics in class_report.items():
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"class_{class_label}_precision", metrics['precision'])
                    mlflow.log_metric(f"class_{class_label}_recall", metrics['recall'])
                    mlflow.log_metric(f"class_{class_label}_f1", metrics['f1-score'])
                    mlflow.log_metric(f"class_{class_label}_support", metrics['support'])
        
        cm_flat = cm.flatten()
        for idx, value in enumerate(cm_flat):
            mlflow.log_metric(f"confusion_matrix_pos_{idx}", value)
        
        # save confution metric image in mlflow artifact
        with tempfile.TemporaryDirectory() as tmpdir:

            tmp_path = os.path.join(tmpdir, 'confusion_matrix.png')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=np.unique(y), yticklabels=np.unique(y))
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(tmp_path, format='png')
            plt.close()
            
            mlflow.log_artifact(tmp_path, "confusion_matrix.png")
        
        # Save and log scaler
        # in mlflow as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, "preprocessor")
        
        mlflow.sklearn.log_model(current_best_model, "model")
        
        # Store run_id for best model tracking
        current_run_id = mlflow.active_run().info.run_id

    if val_roc > best_overall_score:
        best_overall_score = val_roc
        best_overall_model = current_best_model
        best_overall_name = name
        best_overall_run_id = current_run_id

print(f"Best Overall Model: {best_overall_name}")
print(f"Best ROC-AUC Score: {best_overall_score:.4f}")



model_name = "diabetes-prediction-model"

try:
    # log and register the best model
    with mlflow.start_run(run_name=f"best_model_{best_overall_name}") as run:

        mlflow.log_param("best_model_name", best_overall_name)
        mlflow.log_param("selection_metric", "val_roc_auc")
        mlflow.log_metric("best_val_roc_auc", best_overall_score)
        
        # register the model
        model_uri = f"runs:/{best_overall_run_id}/model"
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"✓ Model registered !")
        print(f"  - Model Name: {model_name}")
        print(f"  - Version: {model_details.version}")
        print(f"  - Run ID: {best_overall_run_id}")
        
        # Promote model to Production stage
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✓ Model promoted to Production stage!")
        print(f"  - Stage: Production")
        print(f"  - Previous versions archived")
        
except Exception as e:
    print(f"Failed to register or promote model: {e}")
