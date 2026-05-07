from fastapi import FastAPI
import pandas as pd
import joblib

from api.preprocess import preprocess 
from utils.path import get_model_path

app = FastAPI()

model = joblib.load(get_model_path("xgb_model.joblib"))
scaler = joblib.load(get_model_path("scaler.joblib"))
feature_columns = joblib.load(get_model_path("feature_columns.joblib"))

@app.get("/")
def home():
    return {"message": "app is running"}

@app.post("/predict")
def predict(data: dict):

    #JSON to DataFrame
    pd = pd.DataFrame([data])

    #preprocessing
    df = preprocess(df)

    #feature alognment
    df = df[feature_columns]

    #scaling
    df = scaler.transform(df)

    #prediction
    prediction = model.predict(df)
    proba = model.predict_proba(df)

    #final result
    result = "Yes" if prediction[0] == 1 else "No"
    confidence = float(max(proba[0]))


    return {
        "churn_prediction" : int(prediction[0]),
        "result": result,
        "confidence": round(confidence, 4)

    }