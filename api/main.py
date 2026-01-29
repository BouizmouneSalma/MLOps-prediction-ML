from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import sys
from pathlib import Path

# Add api directory to path
sys.path.insert(0, str(Path(__file__).parent))
# from model_loader import load_best_model
import logging

from model_loader import load_best_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# load model, scaler, and metadata
try:
    model, scaler, model_metadata = load_best_model()
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model, scaler, model_metadata = None, None, None


class PredictionRequest(BaseModel):
    Pregnancies: float = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=70, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    
    def to_feature_array(self) -> List[float]:
        """Convert to ordered feature array for model input"""
        return [
            self.Pregnancies,
            self.Glucose,
            self.BloodPressure,
            self.SkinThickness,
            self.Insulin,
            self.BMI,
            self.DiabetesPedigreeFunction,
            self.Age
        ]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_info: dict


@app.get("/")
def health_check():
    if model is None or scaler is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "scaler_loaded": False,
            "error": "Model or scaler not loaded"
        }    
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "model_name": model_metadata.get("model_type", "Unknown"),
        "model_version": model_metadata.get("model_version", "Unknown"),
        "run_id": model_metadata.get("run_id", "Unknown"),
        "val_roc_auc": model_metadata.get("val_roc_auc")
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    
    try:
        # Convert to feature array
        features = request.to_feature_array()
        X_input = np.array([features])
        
        # Scale and predict
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        logger.info(f"Prediction made: {prediction} (probability: {probability:.4f})")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_info={
                "model_type": model_metadata.get("model_type"),
                "model_version": model_metadata.get("model_version")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_metadata.get("model_type"),
        "model_version": model_metadata.get("model_version"),
        "run_id": model_metadata.get("run_id"),
        "metrics": {
            "val_roc_auc": model_metadata.get("val_roc_auc")
        },
        "input_features": [
            "Pregnancies",
            "Glucose", 
            "BloodPressure",
            "SkinThickness", 
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ],
        "feature_constraints": {
            "Pregnancies": "≥ 0",
            "Glucose": "0-300 mg/dL",
            "BloodPressure": "0-200 mm Hg",
            "SkinThickness": "0-100 mm",
            "Insulin": "0-900 mu U/ml",
            "BMI": "0-70 kg/m²",
            "DiabetesPedigreeFunction": "0-3",
            "Age": "0-120 years"
        }
    }

