
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('salary_detect.csv')


# In[6]:


#df.info
df


# In[4]:


df.head()


# In[7]:


x=df.iloc[:,1:2].values
x


# In[8]:


y=df.iloc[:,2:3].values
y


# In[11]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[12]:


trained=reg.fit(x,y)


# In[14]:


plt.scatter(x,y)
plt.plot(x,trained.predict(x),color='red')


# In[15]:


from sklearn.preprocessing import PolynomialFeatures


# In[16]:


prgr=PolynomialFeatures(degree=4)


# In[17]:


mydata=prgr.fit_transform(x)


# In[19]:


lgr1=LinearRegression()


# In[21]:


trained1=lgr1.fit(mydata,y)


# In[22]:


plt.scatter(x,y)
plt.plot(x,trained1.predict(mydata),color='red')

