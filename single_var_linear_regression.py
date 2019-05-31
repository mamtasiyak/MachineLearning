
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


df=pd.read_csv('salary.csv')


# In[7]:


df.info()
df.head()


# In[8]:


sb.countplot(df['YearsExperience'])    #number


# In[26]:


#data with x axis
x=df.iloc[:,:1].values
x


# In[27]:


#salary putting in y axis
y=df.iloc[:,-1].values
y


# In[61]:


#training and testing set
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.1,random_state=0)


# In[62]:


#calling linear regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression()


# In[63]:


#now applying training data and answer
trained=regression.fit(train_x,train_y)


# In[64]:


prediction=trained.predict(test_x)


# In[65]:


#we arw making model of already trained data
plt.scatter(train_x,train_y,label='exp vs sal',color='magenta')
plt.plot(train_x,trained.predict(train_x),color='cyan')
plt.xlabel('Experience',color='cyan')
plt.ylabel('salary',color='magenta')
plt.legend()
plt.show()


# In[66]:


#we arw making model of already trained data
plt.scatter(train_x,train_y,label='exp vs sal',color='magenta')
plt.plot(test_x,prediction,color='cyan')
plt.xlabel('Experience',color='cyan')
plt.ylabel('salary',color='magenta')
plt.legend()
plt.show()

