#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score,
    f1_score, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# In[7]:


scaler = joblib.load("../Models/scaler.joblib")
model = joblib.load("../Models/xgb_model.joblib")
feature_columns = joblib.load("../Models/feature_columns.joblib")


# In[8]:


df = pd.read_csv("../Data/preprocessed.csv")
df.head()


# In[12]:


x = df.drop(['Churn'], axis=1)
y = df['Churn']

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

xtest = xtest[feature_columns]


# In[14]:


scaled_scaled = xtest.copy()
continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Average']

scaled_scaled[continuous_cols] = scaler.transform(scaled_scaled[continuous_cols])


# In[15]:


y_pred = model.predict(xtest)
y_pred_proba = model.predict_proba(xtest)[:, 1]


# In[19]:


accuracy = accuracy_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)
precision = precision_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)
roc_auc = roc_auc_score(ytest, y_pred_proba)
conf_matrix = confusion_matrix(ytest, y_pred)


# In[20]:


evaluation_result = pd.DataFrame([{
    "Model": "XGB Classifier",  # Name of the model
    "Accuracy": accuracy,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1,
    "ROC AUC Score": roc_auc,
    "Confusion Matrix": str(conf_matrix)  # Convert to string for clean display
}])

evaluation_result


# In[23]:


plt.figure(figsize=(5,4))
labels=['No', 'Yes']
sns.heatmap(
    conf_matrix, annot=True, fmt='g', cmap='Greens', 
    xticklabels=labels, yticklabels=labels
)
plt.title('XGB Classifier Confusion')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[27]:


fpr, tpr, _ = roc_curve(ytest, y_pred_proba)

plt.plot(fpr, tpr, color="#CA0A0A", label= f"XGB Classifier (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="#08570C", linestyle="--", label="Random Guessing (Area = 0.5)")

plt.title("XGB Classifier ROC Curve")
plt.xlabel("False Positive Rate")   
plt.ylabel("True Positive Rate")      
plt.legend(loc='lower right')        
plt.grid(True)                     
plt.show() 

