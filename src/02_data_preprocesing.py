#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import seaborn as sns


# In[16]:


df = pd.read_csv("../Data/cleaned_data.csv")
df.head()


# # Label encoding for binary columns

# In[17]:


df.columns


# In[18]:


df[df['tenure'] == 0]


# In[19]:


df['Average'] = df['TotalCharges'] / df['tenure']


# In[20]:


binary_col = [
    'gender', 
    'Partner', 
    'Dependents', 
    'PhoneService', 
    'MultipleLines',
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection',
    'TechSupport',
    'StreamingTV', 
    'StreamingMovies',
    'PaperlessBilling',
    'Churn'
]

for col in binary_col:
    df[col] = df[col].map({'Male':1, 'Female':0}) if col == 'gender' else df[col].map({'Yes':1, 'No':0})


# #  Ordianl encoding for "Contract column"

# In[21]:


con_map = {
    'Month-to-month':0,
    'One year':1,
    'Two year':2
}
df['Contract'] = df['Contract'].map(con_map)


# # One_hot encoding for multi-cetagory columns

# In[22]:


InternetService_dummy = pd.get_dummies(
    df['InternetService'], 
    prefix='InternetService', 
    drop_first=True
).astype('int')


# In[23]:


PaymentMethod_dummy = pd.get_dummies(
    df['PaymentMethod'], 
    prefix='PaymentMethod', 
    drop_first=True,
    dummy_na=False
).astype('int')


# In[24]:


df = pd.concat([df, InternetService_dummy, PaymentMethod_dummy], axis=1)


# In[25]:


df = df.drop(['InternetService', 'PaymentMethod'], axis=1)


# In[26]:


df.head()


# In[27]:


df.to_csv("../Data/preprocessed.csv", index=False)

