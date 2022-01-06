# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:37:07 2020

@author: Andres
"""

# Plantilla de regresión

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Dividir el data set en conjunto de entrenamiento y conjunto de test
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""

# Ajustar la regresión con el dataset



# Predicción de nuestros Modelos
Y_pred = regression.predict([[6.5]])



# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regression.predict((X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()