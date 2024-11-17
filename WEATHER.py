#!/usr/bin/env python
# coding: utf-8

# ### WE ARE ANALYSING WEATHER PREDICTION
# WE ARE GONNA IMPORT BASIC LIBRARY

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import dtale


# ### WE ARE GONNA LOAD AND READ THE DATASET

# In[2]:


df=pd.read_csv(r"C:\Users\Kiran gowda.A\Downloads\weather[1]\weather_prediction_dataset.csv")
df.head()


# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.isnull().sum()


# ### WE ARE ANALYSING THE PRIMARY DATASET

# In[7]:


dff=pd.read_csv(r"C:\Users\Kiran gowda.A\Downloads\weather[1]\weather_prediction_dataset.csv")
dff.head()


# In[8]:


dff.columns


# In[9]:


df.isnull().sum()


# In[10]:


dff.shape


# In[11]:


dff.size


# In[12]:


dff.dtypes


# In[13]:


dff.describe()


# In[16]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()


# In[17]:


data_encoded=pd.get_dummies(dff,drop_first=True)


# In[18]:


dff=dff.astype(int)


# In[19]:


from sklearn.preprocessing import LabelEncoder
for col in dff.select_dtypes(include='object').columns:
    le=LabelEncoder()
    data[col]=le.fit_transform(data[col])


# In[22]:


print(df.info())


# In[23]:


from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
dff_minmax=pd.DataFrame(min_max.fit_transform(dff),columns=dff.columns)


# In[24]:


dff_minmax.head()


# In[30]:


from sklearn.preprocessing import StandardScaler


# In[31]:


scalar=StandardScaler()
dff_standardized=scalar.fit_transform(dff)
dff_standardized=pd.DataFrame(dff_standardized,columns=dff.columns)


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 


# In[49]:


X = dff.drop('TOURS_temp_max', axis=1)  
Y = dff['TOURS_temp_max'] 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# In[54]:


from sklearn.ensemble import RandomForestRegressor 


# In[58]:


Model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
Model_rf.fit(X_train, y_train)


# In[68]:


Y_pred_rf =Model_rf.predict(X_test)


# In[71]:


print('Mean Absolute Error:', mean_absolute_error(y_test, Y_pred_rf))
print('Mean Squared Error:', mean_squared_error(y_test, Y_pred_rf))
print('R² Score:', r2_score(y_test, Y_pred_rf))


# In[ ]:





# In[ ]:




