
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#loading data
df=pd.read_csv('social.csv')


# In[3]:


# information or metadata of pandas
df.info()


# In[6]:


# installing seaborn
#!pip install seaborn
import seaborn as sb
df.head()


# In[8]:


sb.countplot(df['Gender'])


# In[9]:


sb.countplot(df['Age'])


# In[11]:


#features and labels
features=df.iloc[:,2:4].values
label=df.iloc[:,-1].values


# In[26]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(features,label,test_size=.2,random_state=0)


# In[27]:


#Now time for feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[28]:


train_x=sc.fit_transform(train_x)


# In[29]:


test_x=sc.transform(test_x)


# In[30]:


#calling randome forest    20 trees by default  10 trees
clf=RandomForestClassifier(n_estimators=20,criterion='entropy')


# In[31]:


trained=clf.fit(train_x,train_y)


# In[32]:


prediction=trained.predict(test_x)


# In[34]:


from sklearn.metrics import accuracy_score
accuracy_score(test_y,prediction)

