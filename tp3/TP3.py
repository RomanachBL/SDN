from sklearn import *
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target

from sklearn.neighbors import KNeighborsClassifier
nbors = KNeighborsClassifier(n_neighbors=1)
nbors.fit(X, Y)
i=0
while i<len(X):
    res = nbors.predict(X[i,:].reshape(1, -1))
    '''print (res)'''
    i=i+1

'''err = sum(res != iris.target)
print("Nb erreurs:", err)
print( "Pourcentage de prédiction juste:", (150-err)*100/150)   # % de réussite'''
