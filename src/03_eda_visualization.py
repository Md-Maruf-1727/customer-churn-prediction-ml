#!/usr/bin/env python
# coding: utf-8

# In[222]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[223]:


df = pd.read_csv("../Data/cleaned_data.csv")
df.head()


# In[224]:


df_enocoded = pd.read_csv("../Data/preprocessed.csv")


# In[225]:


df_enocoded.head()


# In[226]:


print(f'Total Customer:{df.shape[0]}')


# In[227]:


churn_percent = df['Churn'].value_counts(normalize=True)*100

plt.figure(figsize=(4,3))
sns.set_style("whitegrid")
sns.countplot(
    df, x='Churn',
    palette="husl",
    width=0.6,
    legend="full"
)
plt.title("Churn Distribution", weight='bold')
plt.xlabel('Churn', weight='bold')
plt.ylabel('Total Customer', weight='bold')
for i, v in enumerate(churn_percent.values):
    plt.text(i, v+1, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=12)
plt.show()


# In[228]:


def churn_count_plot(df, category_col, churn_col='Churn', palette='viridis', ax=None):
    sns.set_style("whitegrid")
    sns.countplot(
        data=df, 
        x=category_col,
        hue=churn_col,
        palette=palette, 
        ax=ax
    )
    ax.set_title(f"{category_col} Count by Churn", weight='bold')
    ax.set_xlabel(category_col, weight='bold')
    ax.set_ylabel("Total Customer", weight='bold')
    for container in ax.containers:
        ax.bar_label(container)

def churn_rate_plot(df, category_col, churn_col='Churn', palette='viridis', ax=None):
    sns.set_style("whitegrid")
    churn_rate = pd.crosstab(df[category_col], df[churn_col], normalize=True) * 100
    churn_rate_yes = churn_rate['Yes'].sort_values(ascending=False)

    sns.barplot(
        x=churn_rate_yes.index, 
        y=churn_rate_yes.values, 
        palette=palette, 
        ax=ax
    )
    ax.set_title(f"{category_col} Churn Rate (%)", weight='bold')
    ax.set_xlabel(category_col, weight='bold')
    ax.set_ylabel("Churn Rate (%)", weight='bold')

    for i, value in enumerate(churn_rate_yes.values):
        ax.text(i, value + 0.3, f"{value:.2f}%", ha='center', va='bottom', weight='bold')


# In[229]:


"""fig, axes = plt.subplots(1, 2, figsize=(10, 5))
churn_count_plot(df, category_col='gender', ax=axes[0])
churn_rate_plot(df, category_col='gender',  ax=axes[1])
plt.tight_layout()
plt.show()"""


# In[230]:


"""df['SeniorCitizen_label'] = df['SeniorCitizen'].map({0: 'No', 1:'Yes'})
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
churn_count_plot(df, category_col='SeniorCitizen', ax=axes[0], palette='Reds')
churn_rate_plot(df, category_col='SeniorCitizen',  ax=axes[1], palette='Reds')
plt.tight_layout()
plt.show()"""


# In[231]:


"""fig, axes = plt.subplots(2, 2, figsize=(10, 10))
churn_count_plot(df, category_col='Partner', ax=axes[0, 0], palette='inferno')
churn_rate_plot(df, category_col='Partner',  ax=axes[0, 1], palette='inferno')

churn_count_plot(df, category_col='Dependents', ax=axes[1, 0], palette='hls')
churn_rate_plot(df, category_col='Dependents',  ax=axes[1, 1], palette='hls')

fig.suptitle("Household / Family Status Analysis vs Customer Churn", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()"""


# In[232]:


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
churn_count_plot(df, category_col='Contract', ax=axes[0], palette='Oranges')
churn_rate_plot(df, category_col='Contract',  ax=axes[1], palette='Oranges')
plt.tight_layout()
plt.show()


# In[233]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
churn_count_plot(df, category_col='PaymentMethod', ax=axes[0], palette='crest')
churn_rate_plot(df, category_col='PaymentMethod',  ax=axes[1], palette='crest')
plt.tight_layout()
plt.show()


# In[234]:


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
churn_count_plot(df, category_col='PaperlessBilling', ax=axes[0], palette='cividis')
churn_rate_plot(df, category_col='PaperlessBilling',  ax=axes[1], palette='cividis')
plt.tight_layout()
plt.show()


# In[235]:


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
churn_count_plot(df, category_col='InternetService', ax=axes[0], palette='Accent')
churn_rate_plot(df, category_col='InternetService',  ax=axes[1], palette='Accent')
plt.tight_layout()
plt.show()


# In[236]:


fig, axes = plt.subplots(4, 2, figsize=(12, 20))
churn_count_plot(df, category_col='OnlineSecurity', ax=axes[0, 0], palette='Spectral')
churn_rate_plot(df, category_col='OnlineSecurity',  ax=axes[0, 1], palette='Spectral')

churn_count_plot(df, category_col='TechSupport', ax=axes[1, 0], palette='Pastel1')
churn_rate_plot(df, category_col='TechSupport',  ax=axes[1, 1], palette='Pastel1')

churn_count_plot(df, category_col='OnlineBackup', ax=axes[2, 0], palette='Purples')
churn_rate_plot(df, category_col='OnlineBackup',  ax=axes[2, 1], palette='Purples')

churn_count_plot(df, category_col='DeviceProtection', ax=axes[3, 0], palette='Greys')
churn_rate_plot(df, category_col='DeviceProtection',  ax=axes[3, 1], palette='Greys')

fig.suptitle("Impact of Online Security & Support Services on Customer Churn", fontsize=18, weight='bold')
plt.tight_layout(pad=2)
plt.show()


# In[237]:


"""fig, axes = plt.subplots(2, 2, figsize=(10, 10))
churn_count_plot(df, category_col='PhoneService', ax=axes[0, 0], palette='PiYG')
churn_rate_plot(df, category_col='PhoneService',  ax=axes[0, 1], palette='PiYG')

churn_count_plot(df, category_col='MultipleLines', ax=axes[1, 0], palette='PRGn')
churn_rate_plot(df, category_col='MultipleLines',  ax=axes[1, 1], palette='PRGn')

plt.tight_layout()
plt.show()"""


# In[238]:


"""fig, axes = plt.subplots(2, 2, figsize=(10, 10))
churn_count_plot(df, category_col='StreamingTV', ax=axes[0, 0], palette='rocket')
churn_rate_plot(df, category_col='StreamingTV',  ax=axes[0, 1], palette='rocket')

churn_count_plot(df, category_col='StreamingMovies', ax=axes[1, 0], palette='flare')
churn_rate_plot(df, category_col='StreamingMovies',  ax=axes[1, 1], palette='flare')

plt.tight_layout()
plt.show()"""


# In[239]:


def hist_plot(df, dist_col, palette='viridis'):
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df,
        x=dist_col,
        hue='Churn',
        kde=True,
        bins=40,
        palette=palette
    )
    plt.title(f"{dist_col} Distribution by churn", fontsize=18, weight='bold')
    plt.xlabel(dist_col, weight='bold')
    plt.ylabel('Number of Customer')
    plt.show()


# In[240]:


hist_plot(df, dist_col='tenure', palette=['blue', 'red'])


# In[241]:


hist_plot(df, dist_col='MonthlyCharges', palette=['green', 'yellow'])


# In[242]:


hist_plot(df, dist_col='TotalCharges', palette=['blue', 'purple'])


# In[243]:


plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x= 'tenure',
    y='MonthlyCharges',
    hue='Churn',
    palette={'No': 'green', 'Yes': 'Red'},
    s= 50
)

plt.title('Tenure vs Monthly Charges by customer Churn', weight='bold')
plt.xlabel('Tenure', weight='bold')
plt.ylabel('Monthly Charges', weight='bold')
plt.show()


# In[244]:


plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x= 'tenure',
    y='TotalCharges',
    hue='Churn',
    palette={'No': 'blue', 'Yes': 'Red'},
    s= 50
)

plt.title('Tenure vs Monthly Charges by customer Churn', weight='bold')
plt.xlabel('Tenure', weight='bold')
plt.ylabel('Monthly Charges', weight='bold')
plt.show()


# In[245]:


plt.figure(figsize=(18, 12))
sns.heatmap(
    df_enocoded.corr(),
    annot=True,
    fmt='.2f',
    cmap='Greens',
    linewidths=0.5,
    linecolor="#0EE431"
)
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.show()


# In[249]:


fig, axes = plt.subplots(1, 5, figsize=(30, 6))
churn_rate_plot(df, category_col='Contract',  ax=axes[0], palette='Spectral')

churn_rate_plot(df, category_col='PaymentMethod',  ax=axes[1], palette='Pastel1')

churn_rate_plot(df, category_col='InternetService',  ax=axes[2], palette='Purples')

churn_rate_plot(df, category_col='OnlineSecurity',  ax=axes[3], palette='Greys')

churn_rate_plot(df, category_col='TechSupport',  ax=axes[4], palette='Greys')

fig.suptitle("Churn Rate Comparison", fontsize=18, weight='bold')
#plt.tight_layout(pad=2)
plt.show()

