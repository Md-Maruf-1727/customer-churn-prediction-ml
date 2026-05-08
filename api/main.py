from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
async def read_index():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(data: ChurnInput):

    # JSON → DataFrame
    df = pd.DataFrame([data.dict()])

    # preprocessing
    df = preprocess(df)

    # ONLY THIS (important)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # scaling
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Average']
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    # prediction
    prediction = model.predict(df)
    proba = model.predict_proba(df)

    result = "Yes (Churn)" if prediction[0] == 1 else "No (Not Churn)"
    confidence = float(max(proba[0]))

    return {
        "churn_prediction": int(prediction[0]),
        "result": result,
        "confidence": round(confidence, 4)
    }