# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:16:02 2020

@author: Andres
"""

# Upper Confidence Bound (UCB)

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Cargar dataset
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

# Algoritmo de Upper Confidence Bound (UCB)
N = 10000
d = 10
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0
for a in range (0, N):
    max_upper_bound = 0
    ad = 0
    for i in range (0, d):
        if (number_of_selections[i]>0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(a+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
        ads_selected.append(ad)
        number_of_selections[ad] = number_of_selections[ad] + 1
        reward = dataset.values[a, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward

# Histogramas de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID de anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()