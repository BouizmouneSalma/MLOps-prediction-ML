"""
Script d'entraînement et de validation du modèle pour le pipeline CI/CD
Entraîne le modèle et valide ses performances
"""
import sys
import json
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.models import get_models_and_grids
from training.data_validation import validate_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def main():
    """Main training and validation function"""
    print("=" * 60)
    print("MODEL TRAINING & VALIDATION PIPELINE")
    print("=" * 60)
    
    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("CI_Pipeline_Training")
    
    # Load and validate data
    data_path = Path(__file__).parent.parent / "data" / "data.csv"
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate dataset
    is_valid, _ = validate_dataset(df, target_column="Cluster")
    if not is_valid:
        print("❌ Data validation failed")
        sys.exit(1)
    
    # Prepare data
    X = df.drop("Cluster", axis=1)
    y = df["Cluster"]
    
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get models
    models_and_grids = get_models_and_grids()
    
    print(f"\n🤖 Training {len(models_and_grids)} models...")
    
    best_model = None
    best_score = 0
    best_metrics = {}
    
    # Train models
    for model_name, (model, param_grid) in models_and_grids.items():
        print(f"\n  Training {model_name}...")
        
        with mlflow.start_run(run_name=f"CI_{model_name}"):
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
            
            # Log to MLflow
            mlflow.log_params({"model_type": model_name})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = model_name
                best_metrics = metrics
    
    # Create reports directory
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Save metrics report
    report = {
        "best_model": best_model,
        "best_score": best_score,
        "metrics": best_metrics,
        "all_models_trained": list(models_and_grids.keys())
    }
    
    report_path = reports_dir / "model_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Metrics report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best Model: {best_model}")
    print(f"Best F1 Score: {best_score:.4f}")
    print("\nBest Model Metrics:")
    for metric, value in best_metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
