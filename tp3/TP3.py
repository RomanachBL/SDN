from sklearn import *
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
x = iris.data
y = iris.target


def PPV(x,y):
    tab = []
    dist = euclidean_distances(x)
    for i, point in enumerate(dist):
        tab.append(y[np.argmin(np.delete(point, i))])
    print ("Pourcentage de réussite = ",str((sum(y == tab)*100)/len(y)),"%")
    print ("Pourcentage d'erreurs = ",str(100-(sum(y == tab)*100)/len(y)),"%")
    return tab


test_ppv = PPV(x,y)
print (test_ppv)


def KPP(x, y, voisin):
    tab = []
    neigh = KNeighborsClassifier(n_neighbors=voisin)
    neigh.fit(x, y)
    for data in x:
        tab.append(neigh.predict([data])[0])
    print ("Pourcentage de réussite = ",str((sum(y == tab)*100)/len(y)),"%")
    print ("Pourcentage d'erreurs = ",str(100-(sum(y == tab)*100)/len(y)),"%")
    return tab

test_kpp1 = KPP(x, y, 1)
print (test_kpp1)
test_kpp2 = KPP(x, y, 2)
print (test_kpp2)
test_kpp3 = KPP(x, y, 3)
print (test_kpp3)





def barycentre(x, y, classe):
    return x[np.where(np.column_stack((x, y)) [:, 4]== classe)].mean(0)

def lesBarycentres(x, y):
    tab = []
    for classe in np.unique(y):
        tab.append(barycentre(x,y,classe))
    return tab

def proba_classe(x0, x, y, classe):
    return 1-(euclidean_distances([x0], [barycentre(x, y, classe)])/euclidean_distances([x0], [lesBarycentres(x, y)]).sum(1))

def CBN(x, y):
    y1 = []
    for i, data in enumerate(x):
        max = 0
        for k in np.unique(y):
            max = proba_classe(data, np.delete(x, i), y, 0)
            y1.append(0)
            if max < proba_classe(data, np.delete(x, i), y, k):
                max = proba_classe(data, np.delete(x, i), y, k)
                y1.delete(-1)
                y1.append(k)
    return y1





    
    
    
    
    