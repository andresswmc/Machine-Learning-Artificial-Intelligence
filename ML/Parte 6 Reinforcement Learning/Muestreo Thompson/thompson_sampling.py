# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:06:54 2020

@author: Andres
"""

# Muestreo Thompson

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Cargar dataset
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv")

# Algoritmo de Upper Confidence Bound (UCB)
N = 10000
d = 10
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0
for a in range (0, N):
    max_random = 0
    ad = 0
    for i in range (0, d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)           
        if random_beta > max_random:
            max_random = random_beta
            ad = i
        ads_selected.append(ad)
        reward = dataset.values[a, ad]
        if reward == 1:
            number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
        else:
            number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
        total_reward = total_reward + reward

# Histogramas de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID de anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()