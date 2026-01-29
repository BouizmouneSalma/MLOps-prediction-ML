

import mlflow 
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("../data/data.csv")

X = df.drop("Cluster")
y = df["Cluster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")