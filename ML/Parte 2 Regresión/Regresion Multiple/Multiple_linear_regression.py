# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:45:31 2020

@author: Andres
"""

# Regresión Lineal Simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set

dataset = pd.read_csv("G:/export.csv")
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 3].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
X = np.array(ct.fit_transform(X), dtype=np.float)

# Evitar la trampa de las variables ficticias

X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Ajustar el modelo de regresión lineal con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Prediccion de los resultados en el conjunto de testing
Y_pred = regression.predict(X_test)

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

