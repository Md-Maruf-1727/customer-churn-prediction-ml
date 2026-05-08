import streamlit as st
import pandas as pd
import joblib
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


try:
    from api.preprocess import preprocess
    from utils.path import get_model_path
except ImportError as e:
    st.error(f"Error: Could not import project modules. {e}")
    st.stop()


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")


st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Customer Churn Analysis System")
st.write("Fill in the customer information below to predict the probability of churn.")


@st.cache_resource
def load_assets():
    try:
        model = joblib.load(get_model_path("xgb_model.joblib"))
        scaler = joblib.load(get_model_path("scaler.joblib"))
        feature_columns = joblib.load(get_model_path("feature_columns.joblib"))
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model files. Make sure they are in the 'model' directory. Details: {e}")
        return None, None, None

model, scaler, feature_columns = load_assets()

if model is not None:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

        with col2:
            st.subheader("Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("Billing & Contract")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", value=50.0)
            total = st.number_input("Total Charges ($)", value=50.0 * 12)

        submitted = st.form_submit_button("Predict Churn Status")

    if submitted:
        raw_data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet, 
            "OnlineSecurity": security, "OnlineBackup": backup, 
            "DeviceProtection": device, "TechSupport": support,
            "StreamingTV": "No", "StreamingMovies": "No", 
            "Contract": contract, "PaperlessBilling": paperless, 
            "PaymentMethod": payment, "MonthlyCharges": monthly, 
            "TotalCharges": total
        }

        df = pd.DataFrame([raw_data])
        df_processed = preprocess(df)
        
        df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)

        continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Average']
        df_processed[continuous_cols] = scaler.transform(df_processed[continuous_cols])

        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0]

        st.write("---")
        if prediction == 1:
            st.error(f"### Prediction: Customer is likely to CHURN")
            st.metric("Churn Probability", f"{round(probability[1] * 100, 2)}%")
        else:
            st.success(f"### Prediction: Customer is likely to STAY")
            st.metric("Retention Probability", f"{round(probability[0] * 100, 2)}%")
else:
    st.info("System is waiting for model assets...")