#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score,
    precision_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib


# In[2]:


df = pd.read_csv("../Data/preprocessed.csv")
df.head()


# In[3]:


x = df.drop(['Churn'], axis=1)
y = df['Churn']                              


# In[4]:


xtrain, xtest, ytrain, ytest = train_test_split(
    x, y,
    test_size=0.2, 
    random_state=42,                                                                     
    stratify=y
)


# In[5]:


xtrain_no_scale = xtrain.copy()
ytrain_no_scale = ytrain.copy()


# In[6]:


smote = SMOTE(random_state=42)
xtrain_no_scale_res, ytrain_no_scale_res = smote.fit_resample(xtrain_no_scale, ytrain_no_scale)


# # Applying without Scalling

# In[7]:


algorithms = {
    "LogisticRegression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "XGBClassifier": XGBClassifier(random_state=42, eval_metric='logloss'),
    "GaussianNB": GaussianNB()
}


# In[8]:


result_list = []

for model_name, model in algorithms.items():
    model.fit(xtrain_no_scale_res, ytrain_no_scale_res)
    pred = model.predict(xtest)
    pred_proba = model.predict_proba(xtest)[:, 1]

    result_list.append({
        'Model': model_name,
        'Train Score': model.score(xtrain_no_scale_res, ytrain_no_scale_res),
        'Test Score': model.score(xtest, ytest),
        'Accuracy Score': accuracy_score(ytest, pred),
        'Recall Score': recall_score(ytest, pred),
        'Precision Score': precision_score(ytest, pred),
        'F1 Score': f1_score(ytest, pred),
        'Roc Auc Score': roc_auc_score(ytest, pred_proba),
        'Confussion Matrix': confusion_matrix(ytest, pred)

    })

results = pd.DataFrame(result_list)


# In[9]:


print("Result without scaling:")
results.sort_values(by='Roc Auc Score', ascending=False)


# # Apply with Scaling

# In[10]:


scale_xtrain = xtrain.copy()
scale_xtest = xtest.copy()


# In[11]:


continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Average']

scaler = StandardScaler()

scale_xtrain[continuous_cols] = scaler.fit_transform(scale_xtrain[continuous_cols])
scale_xtest[continuous_cols] = scaler.transform(scale_xtest[continuous_cols])


# In[12]:


smote = SMOTE(random_state=42)
scale_xtrain_res, scale_ytrain_res = smote.fit_resample(scale_xtrain, ytrain)


# In[13]:


result_list_scaled = []

for model_name, model in algorithms.items():
    model_sceled = model.fit(scale_xtrain_res, scale_ytrain_res)
    scaled_pred = model_sceled.predict(scale_xtest)
    scaled_pred_proba = model_sceled.predict_proba(scale_xtest)[:, 1]

    result_list_scaled.append({
        'Model': model_name,
        'Train Score': model_sceled.score(scale_xtrain_res, scale_ytrain_res),
        'Test Score': model_sceled.score(scale_xtest, ytest),
        'Accuracy Score': accuracy_score(ytest, scaled_pred),
        'Recall Score': recall_score(ytest, scaled_pred),
        'Precision Score': precision_score(ytest, scaled_pred),
        'F1 Score': f1_score(ytest, scaled_pred),
        'Roc Auc Score': roc_auc_score(ytest, scaled_pred_proba),
        'Confusion Matrix': confusion_matrix(ytest, scaled_pred)
    })

    scaled_results = pd.DataFrame(result_list_scaled)


# In[14]:


print("Result with scaling:")
scaled_results.sort_values(by='Roc Auc Score', ascending=False)


# # Evaluation Function

# In[15]:


def evaluete_model(model, xtrain, xtest, ytrain, ytest, model_name):
    xtrain_pred = model.predict(xtrain)
    y_pred = model.predict(xtest)
    y_pred_proba = model.predict_proba(xtest)[:, 1]

    train_recall = recall_score(ytrain, xtrain_pred)
    test_recall = recall_score(ytest, y_pred)

    accuracy = accuracy_score(ytest, y_pred)  
    recall = recall_score(ytest, y_pred)  
    precision = precision_score(ytest, y_pred)  
    f1 = f1_score(ytest, y_pred)  
    roc_auc = roc_auc_score(ytest, y_pred_proba)
    conf_matrix = confusion_matrix(ytest, y_pred)        

    best_params = model.best_params_ if hasattr(model, 'best_params_') else None
    best_score = model.best_score_ if hasattr(model, 'best_score_') else None

    result = pd.DataFrame([{
        'Model': model_name,
        'Train Recall': train_recall,
        'Test Recall': test_recall,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Confusion Matrix': str(conf_matrix),
        'Roc Auc Score': roc_auc,
        'Best Params': str(best_params),
        'Best CV Score': best_score
    }])

    return result, y_pred_proba, conf_matrix, roc_auc


# # GaussianNB()

# In[16]:


pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', GaussianNB())
])

gnb_param_grid = {
    'model__var_smoothing': [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

gnb_grid = GridSearchCV(
    estimator=pipeline, param_grid=gnb_param_grid, cv = 10, scoring='recall'
)
gnb_model = gnb_grid.fit(scale_xtrain, ytrain)


# In[17]:


gnb_result, gnb_pred_proba, gnb_confusion, gnb_roc_auc = evaluete_model(
    model = gnb_model, 
    xtrain = scale_xtrain, 
    xtest = scale_xtest,
    ytrain = ytrain,
    ytest = ytest,
    model_name = 'GaussianNB'
)
gnb_result


# In[18]:


plt.figure(figsize=(5,4))
labels=['No', 'Yes']
sns.heatmap(
    gnb_confusion, annot=True, fmt='g', cmap='YlGnBu', 
    xticklabels=labels, yticklabels=labels
)
plt.title('GaussianNB Confusion')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Logistic Regression

# In[19]:


logi_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=500))
])

log_param_grid = {
    'model__C': [0.01, 0.05, 0.1, 0.5, 1],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear'],
    'model__class_weight': [None, 'balanced']
}
logistic_grid = GridSearchCV(
    estimator=logi_pipeline,
    param_grid=log_param_grid,
    cv= 5, 
    scoring='f1',
    n_jobs=1
)
logistic_model = logistic_grid.fit(scale_xtrain, ytrain)


# In[20]:


logistic_result, logistic_pred_proba, logistic_confusion, logistic_roc_auc = evaluete_model(
    model=logistic_model,
    xtrain=scale_xtrain, 
    xtest=scale_xtest,
    ytrain=ytrain, 
    ytest=ytest,
    model_name='Logictic Regression'
)
logistic_result


# In[21]:


plt.figure(figsize=(5,4))
labels=['No', 'Yes']
sns.heatmap(
    logistic_confusion, annot=True, fmt='g', cmap='Reds', 
    xticklabels=labels, yticklabels=labels
)
plt.title('Logistic Regression Confusion')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Random Forest Classifier

# In[22]:


rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf':[1, 2], 
    'model__max_features': ['sqrt'],
    'model__class_weight': [None, 'balanced']
}

rf_grid = GridSearchCV(
    estimator= rf_pipeline,
    param_grid=rf_param_grid,
    cv = 5, 
    scoring='recall',
    n_jobs=1,
    verbose=1
)
rf_model = rf_grid.fit(scale_xtrain, ytrain)


# In[23]:


rf_result, rf_pred_proba, rf_confusion, rf_roc_auc = evaluete_model(
    model=rf_model,
    xtrain=scale_xtrain, 
    xtest=scale_xtest,
    ytrain=ytrain, 
    ytest=ytest,
    model_name='Random Forest Classifier'
)
rf_result


# In[24]:


plt.figure(figsize=(5,4))
labels=['No', 'Yes']
sns.heatmap(
    rf_confusion, annot=True, fmt='g', cmap='Blues', 
    xticklabels=labels, yticklabels=labels
)
plt.title('Random Forest Confusion')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # XGBclassifier

# In[25]:


xgb_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(random_state=42, eval_metric='logloss'))
])

xgb_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

xgb_gird = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_param_grid,
    scoring='recall',
    cv = 5,
    n_jobs=1,
    verbose=1
)

xgb_model = xgb_gird.fit(scale_xtrain, ytrain)


# In[26]:


xgb_result, xgb_pred_proba, xgb_confusion, xgb_roc_auc = evaluete_model(
    model=xgb_model,
    xtrain=scale_xtrain, 
    xtest=scale_xtest,
    ytrain=ytrain, 
    ytest=ytest,
    model_name='XGB Classifier'
)
xgb_result


# In[27]:


plt.figure(figsize=(5,4))
labels=['No', 'Yes']
sns.heatmap(
    xgb_confusion, annot=True, fmt='g', cmap='viridis', 
    xticklabels=labels, yticklabels=labels
)
plt.title('XGB Classifier Confusion')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[28]:


comparing = pd.concat([gnb_result, logistic_result, rf_result, xgb_result], ignore_index=True)
comparing.sort_values(by='Recall', ascending=False)


# In[29]:


confussion_matrixes = [
    gnb_confusion, logistic_confusion, 
    rf_confusion, xgb_confusion
]
titles = [
    'GaussianNB', 'Logistic Regression', 
    'Random Forest Classifier', 'XGB Classifier'
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
labels = ['No', 'Yes']

for ax, cm, title in zip(axes.flatten(), confussion_matrixes, titles):
    sns.heatmap( 
        cm, annot=True, fmt='g', cmap='Greens', ax= ax,
        xticklabels=labels, yticklabels=labels
    )
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()


# In[30]:


fpr1, tpr1, _ = roc_curve(ytest, gnb_pred_proba)
fpr2, tpr2, _ = roc_curve(ytest, logistic_pred_proba)
fpr3, tpr3, _ = roc_curve(ytest, rf_pred_proba)
fpr4, tpr4, _ = roc_curve(ytest, xgb_pred_proba)

plt.plot(fpr1, tpr1, color="#E71313", label= f"GaussianNB (AUC = {gnb_roc_auc:.2f})")
plt.plot(fpr2, tpr2, color="#16E62A", label= f"Logistic Regression (AUC = {logistic_roc_auc:.2f})")
plt.plot(fpr3, tpr3, color="#3862EE", label= f"Random Forest (AUC = {rf_roc_auc:.2f})")
plt.plot(fpr4, tpr4, color="#000000", label= f"XGB Classifier (AUC = {xgb_roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color='red', linestyle="--", label="Random Guessing (Area = 0.5)")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")   
plt.ylabel("True Positive Rate")      
plt.legend(loc='lower right')        
plt.grid(True)                     
plt.show()   


# In[31]:


best_rf = xgb_model.best_estimator_.named_steps['model']

feature_importance = pd.DataFrame({
    'Feature' : scale_xtrain.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance


# In[32]:


plt.figure(figsize=(8, 5))
sns.barplot(
    data=feature_importance.head(15), 
    x='Importance', y= 'Feature', 
    palette='Set1'
)
plt.title("Top 15 Important Features in XGB Classifier", weight='bold')
plt.show()


# In[33]:


feature_columns = scale_xtrain.columns.tolist()
feature_columns


# In[34]:


joblib.dump(scaler, "../Models/scaler.joblib")


# In[35]:


joblib.dump(xgb_model, "../Models/xgb_model.joblib")


# In[36]:


joblib.dump(feature_columns, "../Models/feature_columns.joblib")


# In[ ]:




