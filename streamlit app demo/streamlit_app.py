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

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1400px !important; }

/* ── HERO HEADER ── */
.hero-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a1f3a 50%, #0f1629 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(246,135,179,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed 0%, #b794f4 50%, #f687b3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1rem;
    color: #8892a4;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── SECTION CARDS ── */
.section-card {
    background: linear-gradient(145deg, #111827, #0f1520);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    transition: border-color 0.3s ease;
}
.section-card:hover { border-color: rgba(99,179,237,0.2); }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title.blue  { color: #63b3ed; }
.section-title.purple{ color: #b794f4; }
.section-title.pink  { color: #f687b3; }

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #0d1117 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important;
    outline: none !important;
}

/* ── SUBMIT BUTTON ── */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.3) !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.45) !important;
}

/* ── RESULT CARDS ── */
.result-wrapper {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border-color: rgba(239,68,68,0.3);
}
.result-stay {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border-color: rgba(34,197,94,0.3);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-churn .result-label { color: #f87171; }
.result-stay  .result-label { color: #4ade80; }

.result-desc {
    color: #8892a4;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.prob-bar-fill-churn {
    background: linear-gradient(90deg, #ef4444, #f97316);
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}
.prob-bar-fill-stay {
    background: linear-gradient(90deg, #22c55e, #3b82f6);
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}
.prob-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
}
.result-churn .prob-value { color: #f87171; }
.result-stay  .prob-value { color: #4ade80; }

/* ── METRIC CHIPS ── */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 0.78rem;
    color: #8892a4;
}
.metric-chip span {
    color: #e8eaf0;
    font-weight: 600;
    display: block;
    font-size: 1rem;
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.2), rgba(183,148,244,0.2), transparent);
    margin: 1.5rem 0;
}

/* ── STREAMLIT ALERT OVERRIDES ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── HERO ──
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🛡️ AI Powered Analytics</div>
    <div class="hero-title">ChurnGuard AI</div>
    <div class="hero-sub">Predict customer churn probability with machine learning — fill in customer details below.</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    try:
        model        = joblib.load(get_model_path("xgb_model.joblib"))
        scaler       = joblib.load(get_model_path("scaler.joblib"))
        feature_cols = joblib.load(get_model_path("feature_columns.joblib"))
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Error loading model files. Make sure they are in the 'model' directory. Details: {e}")
        return None, None, None

model, scaler, feature_columns = load_assets()

if model is not None:
    with st.form("prediction_form"):

        # ── ROW 1: Demographics ──
        st.markdown('<div class="section-card"><div class="section-title blue">👤 Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        gender     = c1.selectbox("Gender",          ["Male", "Female"])
        senior     = c2.selectbox("Senior Citizen",  ["No", "Yes"])
        partner    = c3.selectbox("Partner",          ["Yes", "No"])
        dependents = c4.selectbox("Dependents",       ["No", "Yes"])
        tenure     = c5.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ── ROW 2: Services ──
        st.markdown('<div class="section-card"><div class="section-title purple">📡 Services</div>', unsafe_allow_html=True)
        s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
        phone    = s1.selectbox("Phone Service",    ["Yes", "No"])
        multiple = s2.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
        internet = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = s4.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        backup   = s5.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
        device   = s6.selectbox("Device Protection",["No", "Yes", "No internet service"])
        support  = s7.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ── ROW 3: Billing ──
        st.markdown('<div class="section-card"><div class="section-title pink">💳 Billing & Contract</div>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)
        contract  = b1.selectbox("Contract Type",    ["Month-to-month", "One year", "Two year"])
        paperless = b2.selectbox("Paperless Billing", ["Yes", "No"])
        payment   = b3.selectbox("Payment Method",   ["Electronic check", "Mailed check",
                                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly   = b4.number_input("Monthly Charges ($)", value=50.0, step=0.01)
        total     = b5.number_input("Total Charges ($)",   value=600.0, step=0.01)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Analyze Churn Risk")

    # ── RESULT ──
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

        df           = pd.DataFrame([raw_data])
        df_processed = preprocess(df)
        df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)

        continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Average']
        df_processed[continuous_cols] = scaler.transform(df_processed[continuous_cols])

        prediction  = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0]

        churn_pct     = round(probability[1] * 100, 1)
        retention_pct = round(probability[0] * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-wrapper result-churn">
                <div class="result-label">⚠️ High Churn Risk</div>
                <div class="result-desc">This customer is likely to leave. Consider a retention campaign.</div>
                <div class="prob-value">{churn_pct}%</div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;text-transform:uppercase;letter-spacing:.06em;">Churn Probability</div>
                <div class="prob-bar-bg" style="margin-top:12px;">
                    <div class="prob-bar-fill-churn" style="width:{churn_pct}%;"></div>
                </div>
                <div class="metric-row">
                    <div class="metric-chip">Retention Chance<span>{retention_pct}%</span></div>
                    <div class="metric-chip">Tenure<span>{tenure} months</span></div>
                    <div class="metric-chip">Monthly Charges<span>${monthly}</span></div>
                    <div class="metric-chip">Contract<span>{contract}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrapper result-stay">
                <div class="result-label">✅ Low Churn Risk</div>
                <div class="result-desc">This customer is likely to stay. Keep up the good service!</div>
                <div class="prob-value">{retention_pct}%</div>
                <div style="color:#8892a4;font-size:0.78rem;margin-top:4px;text-transform:uppercase;letter-spacing:.06em;">Retention Probability</div>
                <div class="prob-bar-bg" style="margin-top:12px;">
                    <div class="prob-bar-fill-stay" style="width:{retention_pct}%;"></div>
                </div>
                <div class="metric-row">
                    <div class="metric-chip">Churn Risk<span>{churn_pct}%</span></div>
                    <div class="metric-chip">Tenure<span>{tenure} months</span></div>
                    <div class="metric-chip">Monthly Charges<span>${monthly}</span></div>
                    <div class="metric-chip">Contract<span>{contract}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("⏳ System is waiting for model assets...")