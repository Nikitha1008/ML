#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[2]:


cars = pd.read_csv("C:/Users/admin/Desktop/nikitha ml/cars.csv", na_values =' ')
cars.head()


# In[3]:


cars.columns =['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',
       'year', 'brand']


# In[4]:


cars =cars.dropna()
cars['cubicinches'] = cars['cubicinches'].astype(int)
cars['weightlbs'] = cars['weightlbs'].astype(int)


# In[5]:


cars.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',
       'year', 'brand']


# In[6]:


cars.columns


# In[7]:


X = cars.iloc[:,:7]


# In[8]:


X.head()


# In[9]:


X.describe()


# In[10]:


X_array = X.values  
X_array


# In[11]:


X.head()


# In[13]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_array)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('metodo cotuvelo - ebow method')
plt.xlabel('numero de clusters')
plt.ylabel('WCSS')
plt.show()


# In[14]:


kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
kmeans.fit_predict(X_array)


# In[15]:


kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
cars['clusters'] = kmeans.fit_predict(X_array)
cars.head()


# In[16]:


kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
cars['clusters'] = kmeans.fit_predict(X_array)
cars.head()
cars.groupby("clusters").agg('mean').plot.bar(figsize=(10,7.5))
plt.title("gastos por clusteer")


# In[18]:


import pandas as pd
cars = pd.read_csv("C:/Users/admin/Desktop/nikitha ml/cars.csv")


# In[ ]:




