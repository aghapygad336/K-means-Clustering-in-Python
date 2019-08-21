# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:54:14 2019

@author: Aghapy
"""
import matplotlib.pyplot as plt
import math  
import numpy as np

def manhattan_distance(pt1, pt2, pt3):
    distance = 0
  
    for i in range(len(pt1)):
        distance += abs(pt1[i]- pt2[i] - pt3[i])

    return distance



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
            

x =[0.5,2.2,3.9,2.1,0.5,0.8,2.7,2.5,2.8,0.1]
y =[4.5,1.5,3.5,1.9,3.2,4.3,1.1,3.5,3.9,4.1]
z =[2.5,0.1,1.1,4.9,1.2,2.6,3.1,2.8,1.5,2.9]
print ('Print 3D')

    #Assuming correct input to the function where the lengths of two features are the same



print("Eculdian Distance")
for i in range(10):
    squared_distance = 0 
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


data = np.array([[0.5,4.5,2.5], [2.2,1.5,0.1], [3.9,3.5,1.1],[2.1,1.9,4.9],[0.5,3.2,1.2],[0.8,4.3,2.6],[2.7,1.1,3.1],[2.5,3.5,2.8],[2.8,3.9,1.5],[0.1,4.1,2.9]])#choose your data
centroids, target = own_kmeans(data, 3)
plt.show()







