# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:16:46 2020

@author: Andres
"""

# Regresión Lineal Simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set

dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Dividir el data set en conjunto de entrenamiento y conjunto de test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state = 0)

#Crear modelo de regresión lineal simple con conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

#Predecir el conjunto de test

Y_pred = regression.predict(X_test)

#visualizar los resultados de entrenamiento
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

#visualizar los resultados de test
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()