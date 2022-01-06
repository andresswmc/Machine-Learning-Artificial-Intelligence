# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:14:51 2020

@author: Andres
"""

# Regresión con Árboles de decisión

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


# Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, Y)

# Predicción de nuestros Modelos SVR
Y_pred = regression.predict([[6.5]])

# Visualización de los resultados del SVR
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

