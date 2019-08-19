# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:54:14 2019

@author: Aghapy
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[0.5,2.2,3.9,2.1,0.5,0.8,2.7,2.5,2.8,0.1]
y =[4.5,1.5,3.5,1.9,3.2,4.3,1.1,3.5,3.9,4.1]
z =[2.5,0.1,1.1,4.9,1.2,2.6,3.1,2.8,1.5,2.9]



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()