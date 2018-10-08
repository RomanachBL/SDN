
''' PARTIE A '''

import numpy as np
from sklearn import *
import matplotlib.pyplot as plt

X = np.array([[1,-1,2],[2,0,0],[0,1,-1]])

print(X)
X.var(0)
X.mean(0)

Y = preprocessing.scale(X)
Y.var(0)
''' la variance est de 1'''
Y.mean(0)
''' la moyenne est de 0'''


''' PARTIE B '''

X2 = np.array([[1,-1,2],[2,0,0],[0,1,-1]])
print(X2)

scaler = preprocessing.MinMaxScaler()
pp = scaler.fit(X2)

print(pp)
print(np.mean(scaler.transform(X2), axis=0))
print(np.min(scaler.transform(X2), axis=0))
print(np.max(scaler.transform(X2), axis=0))

''' PARTIE C '''

iris = datasets.load_iris()
x = iris.data[:, :4]
y = iris.target
plt.scatter(x[:, 0], x[:, 1], c=y)
m = 0
for ligne in range(0, len(iris.feature_names)):
    for colonne in range(0, len(iris.feature_names)):
        if ligne != colonne and ligne <= colonne:
            m = m + 1
            plt.subplot(2,3,m)
            plt.scatter(x[:, ligne], x[:, colonne], c=y, alpha=0.8)
            plt.title("figure Ligne %d Colonne %d" % (ligne, colonne))
plt.show()

''' PARTIE D '''

from sklearn.decomposition import PCA
from sklearn.lda import LDA

pca = PCA(n_components = 2)
irisPCA = pca.fit(iris.data).transform(iris.data)
lda = LDA()
irisLDA = lda.fit(iris.data,iris.target).transform(iris.data)

plt.subplot(1,2,1)
plt.scatter(irisLDA[:,0],irisLDA[:,1],c= iris.target)
plt.subplot(1,2,2)
plt.scatter(irisPCA[:,0],irisPCA[:,1],c= iris.target)
plt.show()



