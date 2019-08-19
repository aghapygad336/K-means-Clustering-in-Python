# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:47:56 2019

@author: Aghapy
"""""
##Sheet4

from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def manhattan_distmtx(X, Y):
    f = np.dot(X.sum(axis=1).reshape(-1, 1), Y.sum(axis=1).reshape(-1, 1).T)
    return f / Y.sum(axis=1) - Y.sum(axis=1)

X = np.array([[2, 2], [1, 14], [10, 7],
              [1, 11], [3, 4], [11, 8],[4,3],[12,9]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

kmeans.predict([[2,2], [1,14],[4,2]])
centroids=kmeans.cluster_centers_


print("Ecludian")
test = euclidean_distances(X, X)
print(test)
print("Manhattan")



manhattan_distmtx(X,X)

