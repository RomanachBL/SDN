from sklearn import *
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


iris = datasets.load_iris()
x = iris.data
y = iris.target


def PPV(x,y):
    tab = []
    dist = euclidean_distances(x)
    for i, point in enumerate(dist):
        tab.append(y[np.argmin(np.delete(point, i))])
    return tab

a = PPV(x,y)
print (a)

print ("Pourcentage de réussite = ",str((sum(y == a)*100)/len(y)),"%")

print ("Pourcentage d'erreurs = ",str(100-(sum(y == a)*100)/len(y)),"%")