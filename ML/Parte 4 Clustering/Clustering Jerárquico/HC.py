# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:33:05 2020

@author: Andres
"""

# Clustering Jerárquico

# Importar las Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar los datos del centro comercial con pandas
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Utilizar el dendograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendograma = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia euclídea")
plt.show()

# Ajustar el clustering jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)

# Visualización de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "brown", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de gastos(1-100)")
plt.legend()
plt.show()