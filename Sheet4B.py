# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:54:14 2019

@author: Aghapy
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import math  






fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[0.5,2.2,3.9,2.1,0.5,0.8,2.7,2.5,2.8,0.1]
y =[4.5,1.5,3.5,1.9,3.2,4.3,1.1,3.5,3.9,4.1]
z =[2.5,0.1,1.1,4.9,1.2,2.6,3.1,2.8,1.5,2.9]
print ('Print 3D')

    #Assuming correct input to the function where the lengths of two features are the same




ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


print("Eculdian Distance")
for i in range(10):
    squared_distance = 0 
    #xD=x[i]–y[i]-z[i]
    aa=x[i]
    bb=y[i]
    cc=z[i]
    xD=aa-bb-cc
    squared_distance +=(xD)**2
    ed = math.sqrt(squared_distance)
    print(ed)


