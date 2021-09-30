#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\Niranjan\Wholesale customers data.csv")


# In[3]:


df


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


X=2*np.random.rand(100,2)


# In[6]:


X1=1+2*np.random.rand(50,2)


# In[7]:


X[50:100,:]=X1


# In[8]:


plt.scatter(X[:,0],X[:,1],s=50,c='g')
plt.show()


# In[9]:


Kmean = KMeans(n_clusters=2)
Kmean.fit(X)


# In[10]:


Kmean.cluster_centers_


# In[11]:


Kmean.labels_


# In[12]:


Kmean.inertia_


# In[13]:


plt.scatter(0.98908685, 0.84609849, s=200, c='r',marker='s')
plt.scatter(2.06198646, 2.12223586,s=200, c='r',marker='s')
plt.scatter(X[:,0],X[:,1],s=50,c='#00FFFF')
plt.show()


# In[14]:


sample_test=np.array([-5.0,-3.0])
second_test=sample_test.reshape(1, -1)
print(Kmean.predict(second_test))


# In[15]:


df.head()


# In[16]:


df.describe()


# In[17]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)


# In[18]:


data_scaled


# In[19]:


#%%
# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[20]:


kmeans.labels_


# In[21]:


kmeans.inertia_


# In[22]:



kmeans.cluster_centers_


# In[23]:


plt.scatter([-0.64104498, -0.05158101,  0.12366094, -0.33628412, -0.42241436,
         0.12449116, -0.43800028, -0.09097771],[ 1.43292407,  0.11529873, -0.27641856,  0.75169392,  0.94422034,
        -0.27827435,  0.97905944,  0.20336194], s=200,marker='s')

plt.show()


# In[24]:


## elbow
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
print(SSE)


# In[25]:


type(df)


# In[26]:


#converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[27]:


plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[28]:


#%%
# k means using 5 clusters and k-means++ initialization
kmeans=KMeans(n_jobs=-1,n_clusters=5,init='k-means++')
kmeans.fit(data_scaled)
predict=kmeans.predict(data_scaled)


# In[29]:


#%%
frame = pd.DataFrame(data_scaled)
frame['cluster'] = predict
frame['cluster'].value_counts()


# In[30]:


kmeans.labels_


# In[31]:


from sklearn.datasets import make_blobs


# In[35]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[36]:


features, true_labels = make_blobs(
   n_samples=200,
   centers=3,
   cluster_std=2.75,
   random_state=42
   )


# In[37]:


features,true_labels


# In[38]:


features.shape


# In[39]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[40]:


kmeans = KMeans(
   init="random",
   n_clusters=3,
   n_init=10,
   max_iter=300,
   random_state=42
   )


# In[41]:


kmeans.fit(scaled_features)


# In[42]:


kmeans.inertia_


# In[43]:


kmeans.cluster_centers_


# In[44]:


kmeans.n_iter_


# In[45]:


kmeans.labels_[:5]


# In[46]:


kmeans_kwargs = {
   "init": "random",
   "n_init": 10,
   "max_iter": 300,
   "random_state": 42,
   }


# In[47]:


# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
     kmeans.fit(scaled_features)
     sse.append(kmeans.inertia_)


# In[48]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[49]:


kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
   )


# In[50]:


kl.elbow


# In[51]:


# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
     kmeans.fit(scaled_features)
     score = silhouette_score(scaled_features, kmeans.labels_)
     silhouette_coefficients.append(score)

