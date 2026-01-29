from fastapi import FastAPI

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")



@app.get("/")
def health_check():
    """Health check endpoint"""

    # add scaller and model
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict():

    # validate request
    # scaller
    # predict
    pass


@app.get("/model-info")
def model_info():
    pass

