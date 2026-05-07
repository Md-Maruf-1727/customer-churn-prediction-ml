from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

from api.preprocess import preprocess
from utils.path import get_model_path
from api.schema import ChurnInput

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(get_model_path("xgb_model.joblib"))
scaler = joblib.load(get_model_path("scaler.joblib"))
feature_columns = joblib.load(get_model_path("feature_columns.joblib"))


@app.get("/")
def home():
    return {"message": "app is running"}


@app.post("/predict")
def predict(data: ChurnInput):

    # JSON → DataFrame
    df = pd.DataFrame([data.dict()])

    # preprocessing
    df = preprocess(df)

    # ONLY THIS (important)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # scaling
    df_scaled = scaler.transform(df)

    # prediction
    prediction = model.predict(df_scaled)
    proba = model.predict_proba(df_scaled)

    result = "Yes (Churn)" if prediction[0] == 1 else "No (Not Churn)"
    confidence = float(max(proba[0]))

    return {
        "churn_prediction": int(prediction[0]),
        "result": result,
        "confidence": round(confidence, 4)
    }