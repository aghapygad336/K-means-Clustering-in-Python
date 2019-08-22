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

def eculdian_d(data):

    print("Eculdian Distance Function")
    for i in range(10):
     squared_distance = 0 
    x, y, z = zip(*data)

    #xD=x[i]–y[i]-z[i]
    for i in range(len(x)):
      squared_distance += (x[i]- y[i] - z[i])**2


    ed = math.sqrt(squared_distance)
    print(ed)

    return ed




def manhattan_distance(data):
    print("Manhatan Distance Function")

    distance = 0
    pt1, pt2, pt3 = zip(*data)

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

    #returns new centroids
    print("move")

    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def own_kmeans(data, k):
    print("own_kmeans")

    c = initialize_centroids(data, k)
    for i in range(0,10):
        new_centroids = move_centroids(data, closest_centroid(data, c), c)
        if np.array_equal(new_centroids,c):
            print(closest_centroid(data, c))
            print("\n")
            print(i)
            print("\n")
            print("best position")
            print(c)
            print("manhattan distance go to")
            print(manhattan_distance(c))
            print("ecludian distance go to")
            print(eculdian_d(c))
            split3d(data)

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
centroids, target = own_kmeans(data, 3)
X_scaled = scale(data, 0, 1)
centroids, target = own_kmeans(X_scaled, 3)














    








