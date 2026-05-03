# 📉 Customer Churn Prediction — Telco Dataset

Predicting which customers are likely to leave a telecom company using machine learning.
Built with real-world data, full preprocessing pipeline, and multiple ML models compared.

---

## 📌 Project Overview

Customer churn means when a customer stops using a company's service.
This project analyzes a Telco dataset of 7,043 customers to find patterns
and build a model that predicts whether a customer will churn or not.

---

## 📊 Dataset

- **Source:** Telco Customer Churn Dataset
- **Rows:** 7,043 customers
- **Columns:** 21 features (gender, contract type, payment method, charges, etc.)
- **Target:** Churn (Yes / No)
- **Churn Rate:** 26.5% churned, 73.5% stayed

---

## 🗂️ Project Structure

```
customer-churn-prediction-ml/
│
├── Data/
│   ├── Telco-Customer-Churn.csv
│   ├── cleaned_data.csv
│   └── preprocessed.csv
│
├── Models/
│   ├── xgb_model.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
│
├── Notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
└── README.md
````

---

## ⚙️ What I Did — Step by Step

### Step 1 — Data Cleaning
- Checked and confirmed no duplicate rows
- Found TotalCharges column stored as text — converted to number
- Dropped 11 rows with null values
- Cleaned inconsistent values:
  - "No phone service" → "No"
  - "No internet service" → "No"
- Dropped customerID column (not useful for modeling)

<!-- ADD IMAGE: churn distribution countplot here -->

### Step 2 — Data Preprocessing
- **Label Encoding** for binary columns (Yes/No → 1/0, Male/Female → 1/0)
- **Ordinal Encoding** for Contract column:
  - Month-to-month → 0, One year → 1, Two year → 2
- **One-Hot Encoding** for InternetService and PaymentMethod
- **Feature Engineering:** Created new column `Average` = TotalCharges / tenure
- **StandardScaler** applied to continuous columns (tenure, MonthlyCharges, TotalCharges, Average)

### Step 3 — Exploratory Data Analysis (EDA)
- Analyzed churn rate across all features
- Key findings:
  - Month-to-month contract customers churn the most
  - Electronic check payment users have highest churn rate
  - Fiber optic internet users churn more than DSL users
  - Customers without OnlineSecurity and TechSupport churn more

<!-- ADD IMAGE: churn rate comparison chart (5 plots side by side) here -->

<!-- ADD IMAGE: correlation heatmap here -->

<!-- ADD IMAGE: tenure distribution by churn here -->

<!-- ADD IMAGE: scatter plot tenure vs monthly charges here -->

### Step 4 — Model Training
- Used **SMOTE** to handle class imbalance (26.5% vs 73.5%)
- Trained and compared **5 models:**

| Model | Accuracy | Recall | ROC AUC |
|---|---|---|---|
| XGBoost | 74.5% | 78.9% | 0.83 |
| Random Forest | 76.2% | 74.6% | 0.83 |
| Logistic Regression | 73.8% | 74.3% | 0.83 |
| GaussianNB | 73.3% | 74.9% | 0.82 |
| KNN | 70.9% | 73.8% | 0.78 |

- Used **GridSearchCV** with cross-validation for hyperparameter tuning on all models

<!-- ADD IMAGE: ROC curve comparison (all 4 models) here -->

<!-- ADD IMAGE: all 4 confusion matrices side by side here -->

### Step 5 — Model Evaluation & Selection
- Selected **XGBoost** as the best model
- Final Results:
  - ✅ Accuracy: 74.5%
  - ✅ Recall: 78.9%
  - ✅ Precision: 51.3%
  - ✅ F1 Score: 62.2%
  - ✅ ROC AUC Score: 0.83

<!-- ADD IMAGE: XGBoost final confusion matrix here -->

<!-- ADD IMAGE: XGBoost ROC curve here -->

### Top 15 Most Important Features

<!-- ADD IMAGE: feature importance bar chart here -->

**Contract type** was the #1 most important factor in predicting churn.

---

## 🛠️ Tools & Libraries

| Category | Tools |
|---|---|
| Language | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Model Saving | Joblib |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Md-Maruf-1727/customer-churn-prediction-ml.git

# 2. Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib

# 3. Open Jupyter Notebook
jupyter notebook

# 4. Run notebooks in order:
# 01 → 02 → 03 → 04 → 05
```

---

## 📈 Key Insights

- Customers on **month-to-month contracts** are most likely to churn
- Customers paying via **electronic check** have the highest churn rate
- Customers **without online security or tech support** churn significantly more
- **New customers** (low tenure) churn more than long-term customers
- **Fiber optic** internet users churn more than DSL users

---

## 👤 Author

**Md. Maruf**
GitHub: [github.com/Md-Maruf-1727](https://github.com/Md-Maruf-1727)
