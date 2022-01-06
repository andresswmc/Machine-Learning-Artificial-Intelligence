# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:07:15 2020

@author: Andres
"""

# Apriori

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('F:/machinelearning-az-master/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,
                min_lift = 3, min_length = 2)

# Visualización de los resultados
results = list(rules)
