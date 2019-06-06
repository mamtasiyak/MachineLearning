
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# In[3]:


df=pd.read_csv('salary_detect.csv')


# In[4]:


x=df.iloc[:,1:2].values
y=df.iloc[:,2:3].values


# In[5]:


random_rgr=RandomForestRegressor(n_estimators=20)


# In[6]:


trained=random_rgr.fit(x,y)


# In[9]:


plt.scatter(x,y,color='green')
plt.plot(x,trained.predict(x),color='yellow')

