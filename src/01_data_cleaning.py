#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[37]:


df = pd.read_csv("../Data/Telco-Customer-Churn.csv")
df.head()


# In[38]:


print(f'Row X Column: {df.shape}')


# In[39]:


df.info()


# In[40]:


df.isna().sum()


# In[41]:


df.duplicated().sum()


# In[42]:


df.columns


# In[43]:


df = df.drop(['customerID'], axis=1)


# In[44]:


#Customer Personal Info
sel_p = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
for col in sel_p:
    print(df[col].value_counts())
    print("--")


# In[45]:


# Basics services
sel_b = ['PhoneService', 'MultipleLines', 'InternetService']
for col in sel_b:
    print(df[col].value_counts())
    print("__")


# In[46]:


# Additional Services
sel_a = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
for col in sel_a:
    print(df[col].value_counts())
    print("--")


# In[47]:


sel_a2 = ['TechSupport','StreamingTV', 'StreamingMovies']
for col in sel_a2:
    print(df[col].value_counts())
    print("--")


# In[48]:


# Contract & Billing Info
sel_i = ['Contract', 'PaperlessBilling', 'PaymentMethod']
for col in sel_i:
    print(df[col].value_counts())
    print("--")


# In[49]:


# Target
plt.figure(figsize=(4,3))
sns.countplot(df['Churn'], color="#28CFCF", edgecolor='black')


# In[50]:


df['Churn'].value_counts(normalize=True)*100


# In[51]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[52]:


df['TotalCharges'].isna().sum()


# In[53]:


df = df.dropna()


# In[54]:


df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')


# In[55]:


sel_col =[
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection', 
    'TechSupport',
    'StreamingTV', 
    'StreamingMovies'
]
for col in sel_col:
    df[col] = df[col].replace('No internet service', 'No')


# In[56]:


df.describe()


# In[57]:


(df['tenure']>=5).value_counts()


# In[58]:


pd.crosstab(df['Contract'], df['Churn'])


# In[59]:


pd.crosstab(df['gender'], df['Churn'])


# In[60]:


pd.crosstab(df['PaymentMethod'], df['Churn'])


# In[62]:


sel_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(15,3))
for ax, col2 in zip(axes.flatten(), sel_col):
    sns.boxplot(
        x=df[col2],
        ax = ax,
        color = "#16E6CA",
        linewidth = 1,
        linecolor = 'black',
        width=0.3,
        notch=True
    )
    ax.set_xlabel(col2, fontweight='bold')
    ax.grid(True, linestyle="-")
plt.tight_layout()
plt.show()


# In[63]:


df.to_csv("../Data/cleaned_data.csv", index=False)

