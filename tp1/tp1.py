# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:07:21 2018

@author: romanach
"""

""" Exercice A """ 

from sklearn import *
import numpy as np 
import matplotlib.pyplot as plt


""" Exercice B """ 

""" 1 """
iris = datasets.load_iris()

""" 2 """
print (iris.data)
print (iris.feature_names)
print (iris.target_names)

""" 3 """
np.column_stack((iris.data, iris.target_names[iris.target]))

""" 4 """
iris.data.mean(0)
iris.data.std(0)
iris.data.min(0)
iris.data.max(0)

""" 5 """ 
iris.data.size
iris.data.shape


""" Exercice C """ 

""" 1 """
mnist = datasets.fetch_mldata('MNIST original')

""" 2 """ 
mnist.data
mnist.data.size
mnist.data.shape
np.column_stack((mnist.data, mnist.target))
mnist.data.mean(0)
mnist.data.std(0)
mnist.data.min(0)
mnist.data.max(0)

np.unique(mnist.target)


""" Exercice D """

""" 1 """
help (datasets.make_blobs)

""" 2 """ 
from sklearn.datasets.samples_generator import make_blobs
x, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)

""" 3 """
plt.figure()
plt.scatter(x[:,0], x[:,1], c=y)
plt.title('Ex.D - Qu°3', loc='center')
plt.xlim((-15,15))
plt.ylim((-15,15))
plt.xlabel('en x')
plt.ylabel('en y')
plt.show()

""" 4 """
u ,v = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
"""plt.figure()
plt.scatter(u[:,0], u[:,1], c=v)
plt.title('titre', loc='center')
plt.xlim((-15,15))
plt.ylim((-15,15))
plt.xlabel('en x')
plt.ylabel('en y')
plt.show()"""

m ,n = make_blobs(n_samples=500, centers=3, n_features=2, random_state=0)
"""plt.figure()
plt.scatter(m[:,0], m[:,1], c=n)
plt.title('titre', loc='center')
plt.xlim((-15,15))
plt.ylim((-15,15))
plt.xlabel('en x')
plt.ylabel('en y')
plt.show()"""

a= np.vstack((u,m))
b= np.hstack((v,n+2))
plt.figure()
plt.scatter(a[:,0], a[:,1], c=b, marker="D")
plt.title('Ex.D - Qu°4', loc='center')
plt.xlabel('en x')
plt.ylabel('en y')
plt.show()


