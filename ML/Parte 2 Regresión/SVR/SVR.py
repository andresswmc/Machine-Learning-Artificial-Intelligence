# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:12:31 2020

@author: Andres
"""

# SVR

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Dividir el data set en conjunto de entrenamiento y conjunto de test
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""
# Escalado de variables

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1,1))


# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, Y)

# Predicción de nuestros Modelos SVR
Y_pred = sc_Y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))

# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()