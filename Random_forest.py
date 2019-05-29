#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[40]:


df=pd.read_csv('social.csv')


# In[41]:


df.head(5)


# In[42]:


df.shape


# In[43]:


features=df.iloc[:,:-1].values
features


# In[77]:


label=df.iloc[:,-1].values
label


# In[78]:


actual_feature=features[:,2:]
actual_feature.shape


# In[79]:


#feature scalling
from sklearn.preprocessing import StandardScaler
f_sc=StandardScaler()


# In[80]:


#now traing and testing data sep
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(actual_feature,label,test_size=.2)


# In[81]:


#here apply feature scalling
final_train=f_sc.fit_transform(train_x)   #always max training data


# In[82]:


#testing data
final_test=f_sc.transform(test_x) #only testing data


# In[83]:


#calling random forest classifier
clf=RandomForestClassifier(n_estimators=30,criterion='entropy')  #here n_estimators=no of trees


# In[85]:


trained=clf.fit(final_train,train_y)


# In[89]:


predicted=trained.predict(final_test)


# In[90]:


from sklearn.metrics import accuracy_score


# In[92]:


acc_score=accuracy_score(test_y,predicted)
acc_score


# In[ ]:




