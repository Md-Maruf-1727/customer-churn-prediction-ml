import pandas as pd

def preprocess(df):

    # -------------------
    # 1. Binary encoding
    # -------------------
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
    df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
    df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})

    # -------------------
    # 2. Special case column
    # -------------------
    df["MultipleLines"] = df["MultipleLines"].replace({
        "No phone service": 0,
        "No": 0,
        "Yes": 1
    })

    # -------------------
    # 3. InternetService encoding
    # -------------------
    df["InternetService_Fiber optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["InternetService_No"] = (df["InternetService"] == "No").astype(int)

    # -------------------
    # 4. PaymentMethod encoding
    # -------------------
    df["PaymentMethod_Electronic check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["PaymentMethod_Mailed check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    df["PaymentMethod_Credit card (automatic)"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)

    # -------------------
    # 5. Drop original categorical columns
    # -------------------
    df = df.drop(["InternetService", "PaymentMethod"], axis=1)

    # -------------------
    # 6. Feature engineering
    # -------------------
    df["Average"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df