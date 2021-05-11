#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Dump a ML model, therefore, we use a pickle format for this
import pickle


# In[2]:


df = pd.read_csv('hiring.csv')
df


# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


# experience

df['experience'].fillna(0, inplace=True)
df


# In[6]:


df.isna().sum()


# In[7]:


df['test_score'].mean()


# In[8]:


# test_score

df['test_score'].fillna(df['test_score'].mean(), inplace= True)
df


# In[9]:


df.isna().sum()


# # Data is clean now

# In[10]:


df


# # Lets separate the independent and dependent features now

# In[13]:


X = df.iloc[ : , :-1 ]
X


# In[14]:


y = df.iloc[ : , -1 ]
y


# # Lets now do Feature Engineering on 'experience' column

# In[15]:


X['experience']


# In[16]:


# Convert the text in the experience column to integer numbers

def convrt(x):
    dict = {
        'two' : 2,
        'three' : 3,
        'five' : 5,
        'seven' : 7,
        'ten' : 10,
        'eleven' : 11,
        0 : 0
    }
    return dict[x]


# In[17]:


X['experience'] = X['experience'].apply( lambda x : convrt(x) )


# In[18]:


X


# In[19]:


X.info()


# # X is ready.

# # Since the dataset is very small, we are not doing traintestsplit. However, this is an obvious step to be applied.

# # Lets call the LinearRegression ML algorithm now.

# In[20]:


# Modeling

from sklearn.linear_model import LinearRegression

lr = LinearRegression()


# In[21]:


# Fit the model

lr.fit( X , y )


# # Prediction Part

# In[22]:


y_pred = lr.predict(X)
y_pred


# In[23]:


y


# # Model Evaluation

# In[24]:


from sklearn.metrics import r2_score
print( r2_score(y_pred,y) )


# In[25]:


X


# # Lets predict on some unseen data now

# In[27]:


lr.predict([[3,9,7]])


# In[28]:


lr.predict([[10,10,10]])


# In[29]:


lr.predict([[10,2,3]])


# # Model Deployment
We need to save our 'lr' model to the local disk as 'model.pkl'
# In[30]:


import pickle

pickle.dump( lr , open('model.pkl','wb') )
# Dump this model by the name "model.pkl" in the systems HDD and while
# doing this, write this file using 'wb' i.e. 'write byter' mode.


# # Lets now try to load the same model 'model.pkl' by reading it from the system and using it for prediction

# In[32]:


ds_c20_model = pickle.load(open('model.pkl','rb'))
# 'rb' means 'read bytes'


# In[33]:


ds_c20_model


# In[34]:


lr.predict([[3,9,7]])


# In[35]:


ds_c20_model.predict([[3,9,7]])


# In[36]:


lr.predict([[10,10,10]])


# In[37]:


ds_c20_model.predict([[10,10,10]])


# In[38]:


lr.predict([[10,2,3]])


# In[39]:


ds_c20_model.predict([[10,2,3]])


# # Happy Learning
