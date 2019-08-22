# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:54:14 2019

@author: Aghapy
"""
import matplotlib.pyplot as plt
import math  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def manhattan_distance(pt1, pt2, pt3):
    distance = 0
  
    for i in range(len(pt1)):
        distance += abs(pt1[i]- pt2[i] - pt3[i])

    return distance



def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom



#points are data points
def initialize_centroids(points, k):
    print("intialize")

    centroids = points.copy()
    np.random.shuffle(centroids)
    print(centroids[:k])
    return centroids[:k]

def closest_centroid(points, centroids):
    print("closest_centroid")

    #return array with index of data to nearest centroid
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    split3d(points)

    #returns new centroids
    print("move")

    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def own_kmeans(data, k):
    print("own_kmeans")

    c = initialize_centroids(data, k)
    for i in range(0, 500):
        new_centroids = move_centroids(data, closest_centroid(data, c), c)
        if np.array_equal(new_centroids,c):
            print(closest_centroid(data, c))
            print("\n")
            print(i)
            print("\n")
            print(c)
            return new_centroids, closest_centroid(data, c)
        else:
            c = new_centroids
            
            
            
def split3d(dataIn):       
    x, y, z = zip(*dataIn)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()   
            
            
            
            


    #Assuming correct input to the function where the lengths of two features are the same
data = np.array([[0.5,4.5,2.5], [2.2,1.5,0.1], [3.9,3.5,1.1],[2.1,1.9,4.9],[0.5,3.2,1.2],[0.8,4.3,2.6],[2.7,1.1,3.1],[2.5,3.5,2.8],[2.8,3.9,1.5],[0.1,4.1,2.9]])#choose your data
split3d(data)
centroids, target = own_kmeans(data, 3)
print("show Target")
print("norlmaization without SKlearn")



X_scaled = scale(data, 0, 1)
split3d(X_scaled)

print(X_scaled)

centroids, target = own_kmeans(X_scaled, 3)
xS, yS, zS = zip(*X_scaled)
print("manhattan_distance")

print(manhattan_distance(xS,yS,zS))


print("3D")










print("Eculdian Distance")
for i in range(10):
    squared_distance = 0 
    x, y, z = zip(*data)

    #xD=x[i]–y[i]-z[i]
    aa=x[i]
    bb=y[i]
    cc=z[i]
    xD=aa-bb-cc
    squared_distance+=(xD)**2
    ed = math.sqrt(squared_distance)
    print(ed)
    
print("Manhattan")
print(manhattan_distance(x,y,z))








