<div align="center">

# 📡 ChurnGuard AI
### Telco Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

> **Predict which customers are about to leave — before they do.**  
> An end-to-end machine learning pipeline with REST API, interactive web UI, and Streamlit dashboard.

---

### 🎬 Live Demo

![ChurnGuard AI Demo](static/demo%20app/api%20connected.gif)

> *Real-time churn prediction via the FastAPI-connected web interface*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Business Insights](#-business-insights)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Demo Apps](#-demo-apps)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

**ChurnGuard AI** is a production-ready machine learning system that predicts whether a telecom customer will churn (leave the service). Built on IBM's Telco Customer Churn dataset (7,032 customers), it combines thorough data analysis with an XGBoost classifier to deliver actionable predictions through both a REST API and interactive dashboards.

**Who is this for?**
- 📊 **Business teams** — Get instant churn risk scores to prioritize retention campaigns
- 🧑‍💻 **Developers** — Integrate predictions into any system via the REST API
- 📈 **Data scientists** — Study a clean, end-to-end ML workflow from raw data to deployment

---

## 💡 Business Insights

> These insights come directly from the exploratory data analysis performed in this project.

### 🔴 High-Risk Customer Profiles

| Factor | Churn Rate | Insight |
|---|---|---|
| **Month-to-month contract** | **23.54%** | Nearly 4× higher risk than annual contracts |
| **Electronic check payment** | **15.23%** | Highest churn among all payment methods |
| **Fiber optic internet** | **18.44%** | Despite being premium, customers churn more |
| **No Online Security** | **22.38%** | Unprotected customers are far more likely to leave |
| **No Tech Support** | **22.17%** | Support absence strongly correlates with churn |
| **Paperless Billing** | **19.91%** | Digitally-engaged customers still churn more |

### 🟢 Retention Signals

| Factor | Churn Rate | Insight |
|---|---|---|
| **Two-year contract** | **0.68%** | Long-term commitment = near-zero churn |
| **One-year contract** | **2.36%** | Still 10× safer than month-to-month |
| **With Online Security** | **4.20%** | Security add-on dramatically reduces churn |
| **With Tech Support** | **4.41%** | Supported customers are 5× less likely to leave |
| **Credit card (auto)** | **3.30%** | Automatic payments correlate with loyalty |

### 🏆 Top Features Driving Churn (XGBoost Feature Importance)

```
Contract Type              ████████████████████████████████  34.0%  ← Most critical
Fiber Optic Internet       ████████████                      12.5%
No Internet Service        ████████                           8.1%
Dependents                 █████                              5.0%
Online Security            █████                              4.7%
Tech Support               ████                               4.2%
Tenure                     ████                               4.2%
```

### 📊 Key Business Numbers
- **26.6%** of customers churn — significant revenue leakage
- **73.4%** customers are retained — strong base to protect
- Customers on **month-to-month contracts** pay higher monthly charges but generate the most churn — a direct trade-off between short-term revenue and long-term retention
- **Gender has virtually no impact** on churn — marketing segmentation by gender alone won't help

### 💼 Recommended Actions
1. **Incentivize long-term contracts** — Offer discounts for annual/biennial commitments to month-to-month customers flagged as high-risk
2. **Bundle security services** — Customers without OnlineSecurity or TechSupport churn at 5× the rate of those who have them
3. **Investigate Fiber Optic quality** — Despite being a premium service, it has the highest churn; pricing or service quality may be the issue
4. **Target electronic check users** — Migrate them to auto-pay methods which correlate with lower churn
5. **Focus on new customers** — Churn is highest in the first few months; a strong onboarding program could significantly improve retention

---

## 📁 Project Structure

```
ChurnGuard-AI/
│
├── 📂 Data/
│   ├── Telco-Customer-Churn.csv       # Raw dataset (7,043 customers, 21 features)
│   ├── cleaned_data.csv               # After null handling & type fixes
│   └── preprocessed.csv              # Encoded & feature-engineered data
│
├── 📂 Models/
│   ├── xgb_model.joblib              # Trained XGBoost model (GridSearchCV optimized)
│   ├── scaler.joblib                 # StandardScaler for continuous features
│   └── feature_columns.joblib        # Ordered feature list for inference
│
├── 📂 NoteBooks/
│   ├── 01_data_cleaning.ipynb        # EDA, null handling, type corrections
│   ├── 02_data_preprocessing.ipynb   # Encoding, feature engineering
│   ├── 03_eda_visualization.ipynb    # Deep visual analysis & business insights
│   ├── 04_model_training.ipynb       # Multi-model training & hyperparameter tuning
│   └── 05_model_evaluation.ipynb     # Final evaluation & ROC analysis
│
├── 📂 api/
│   ├── main.py                       # FastAPI app with /predict endpoint
│   ├── preprocess.py                 # Inference-time preprocessing logic
│   └── schema.py                     # Pydantic input validation schema
│
├── 📂 src/
│   ├── 01_data_cleaning.py           # Script version of notebook 01
│   ├── 02_data_preprocessing.py      # Script version of notebook 02
│   ├── 03_eda_visualization.py       # Script version of notebook 03
│   ├── 04_model_training.py          # Script version of notebook 04
│   └── 05_model_evaluation.py        # Script version of notebook 05
│
├── 📂 static/
│   ├── demo app/
│   │   ├── api connected.gif         # Demo of web UI connected to API
│   │   └── index.html                # Standalone HTML prediction interface
│   └── streamlit_app.py              # ChurnGuard AI dashboard (Streamlit)
│
├── 📂 utils/
│   └── path.py                       # Dynamic path resolution helper
│
├── 📂 Venv/                          # Virtual environment
├── 🐳 Dockerfile                     # Docker container config
├── 📋 requirements.txt               # Python dependencies
└── 📖 README.md                      # You are here
```

---

## 📊 Dataset

**Source:** IBM Telco Customer Churn Dataset  
**Size:** 7,043 customers → 7,032 after cleaning  
**Target:** `Churn` (Yes / No)

| Category | Features |
|---|---|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Billing** | Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| **Engineered** | `Average` (TotalCharges ÷ tenure) |

**Class Distribution:**
```
No Churn  ██████████████████████████████████  73.4%  (5,163 customers)
Churn     ████████████                        26.6%  (1,869 customers)
```
> ⚠️ Class imbalance addressed using **SMOTE** (Synthetic Minority Over-sampling Technique)

---

## 🔧 ML Pipeline

```
Raw Data (CSV)
     │
     ▼
┌─────────────────────────────┐
│  01. Data Cleaning          │  • Convert TotalCharges to numeric
│                             │  • Drop 11 null rows
│                             │  • Remove customerID column
│                             │  • Standardize "No phone/internet service"
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  02. Preprocessing          │  • Label encoding for binary columns
│                             │  • Ordinal encoding for Contract
│                             │  • One-hot encoding for InternetService
│                             │  • One-hot encoding for PaymentMethod
│                             │  • Feature engineering: Average charge/month
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  03. EDA & Visualization    │  • Churn distribution analysis
│                             │  • Correlation heatmap
│                             │  • Feature-wise churn rate plots
│                             │  • Distribution & scatter analysis
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  04. Model Training         │  • Train/test split (80/20, stratified)
│                             │  • SMOTE oversampling on train set
│                             │  • StandardScaler on continuous features
│                             │  • GridSearchCV with 5-fold CV
│                             │  • Models: LR, KNN, RF, XGBoost, GaussianNB
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  05. Model Evaluation       │  • ROC-AUC, Recall, F1, Confusion Matrix
│                             │  • XGBoost selected as best model
│                             │  • Saved: model, scaler, feature_columns
└─────────────┬───────────────┘
              │
              ▼
         FastAPI + Streamlit
```

---

## 📈 Model Performance

### All Models Compared (with Scaling + SMOTE)

| Model | Recall | ROC-AUC | Accuracy | F1 Score |
|---|---|---|---|---|
| 🥇 **XGBoost** | **78.9%** | **0.830** | 74.0% | 0.617 |
| 🥈 Random Forest | 74.6% | 0.830 | 76.2% | 0.625 |
| 🥉 Logistic Regression | 74.3% | 0.829 | 73.8% | 0.601 |
| GaussianNB | 74.9% | 0.818 | 73.3% | 0.598 |
| KNN | 73.8% | 0.779 | 70.9% | 0.574 |

> **Why Recall?** In churn prediction, a missed churner (false negative) costs more than a false alarm. Maximizing Recall ensures we catch the most at-risk customers.

### ✅ Final Model: XGBoost Classifier

```
                  Predicted
                  No    Yes
Actual  No  │  753  │  280  │
        Yes │   79  │  295  │

Accuracy  : 74.5%
Recall    : 78.9%   ← Correctly catches 4 out of 5 churners
Precision : 51.3%
F1 Score  : 62.2%
ROC-AUC   : 0.830   ← Strong discrimination ability
```

**Best Hyperparameters (via GridSearchCV):**
```python
{
  'colsample_bytree': 1.0,
  'learning_rate': 0.1,
  'max_depth': 5,
  'n_estimators': 200,
  'subsample': 0.8
}
```

---

## 🌐 API Reference

### `POST /predict`

Predict churn risk for a single customer.

**Request Body:**
```json
{
  "gender": "Male",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.70,
  "TotalCharges": 848.40
}
```

**Response:**
```json
{
  "churn_prediction": 1,
  "result": "Yes (Churn)",
  "confidence": 0.8732
}
```

| Field | Type | Description |
|---|---|---|
| `churn_prediction` | `int` | `1` = Churn, `0` = No Churn |
| `result` | `string` | Human-readable verdict |
| `confidence` | `float` | Model confidence (0.0 – 1.0) |

### `GET /`

Returns the interactive HTML prediction interface.

---

## 🖥️ Demo Apps

### 1. Streamlit Dashboard (`streamlit_app.py`)
A fully-styled, dark-themed dashboard with form inputs grouped by Demographics, Services, and Billing. Displays churn probability with a progress bar and metric chips.

**Run locally:**
```bash
streamlit run static/streamlit_app.py
```

### 2. HTML Web Interface (`static/demo app/index.html`)
A lightweight, standalone HTML form connected to the FastAPI backend. No framework required — just open in a browser after starting the API.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/churnguard-ai.git
cd churnguard-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks (Optional — models already saved)
```bash
jupyter notebook NoteBooks/
```
Run notebooks `01` through `05` in order to reproduce the full pipeline.

### 5. Start the API
```bash
uvicorn api.main:app --reload --port 8080
```
API docs available at: `http://localhost:8080/docs`

### 6. Launch Streamlit App
```bash
streamlit run static/streamlit_app.py
```

---

## 🐳 Docker Deployment

### Build & Run
```bash
# Build the image
docker build -t churnguard-ai .

# Run the container
docker run -p 8080:8080 churnguard-ai
```

Access the app at: `http://localhost:8080`

The Dockerfile uses `python:3.11-slim`, installs all dependencies from `requirements.txt`, and starts the FastAPI server via `uvicorn` on port `8080`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML / Modeling** | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **Model Serialization** | Joblib |
| **API Framework** | FastAPI + Uvicorn |
| **Data Validation** | Pydantic |
| **Web Dashboard** | Streamlit |
| **Containerization** | Docker |
| **Notebooks** | Jupyter |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for smarter customer retention**

*If this project helped you, please give it a ⭐ on GitHub!*

</div>
