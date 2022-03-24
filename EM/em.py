from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv("EM/xclara.csv")
f1=data['V1'].values
f2=data['V2'].values
X=np.array(list(zip(f1,f2)))

km=KMeans(3)
labels=km.fit(X).predict(X)
centroids=km.cluster_centers_
plt.scatter(X[:,0],X[:,1],c=labels,s=7)
plt.scatter(centroids[:,0],centroids[:,1],c='black',marker='*',s=200)
plt.title('K Means')
plt.show()

em=GaussianMixture(3).fit(X)
labels=em.predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=7)
plt.title('EM')
plt.show()

