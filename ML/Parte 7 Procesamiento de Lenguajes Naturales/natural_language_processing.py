# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:44:36 2020

@author: Andres
"""

# Natural Language Processing

# Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk


# Importar el dataset
dataset = pd.read_csv("F:/machinelearning-az-master/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza de texto
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]"," ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus.append(review)

# Crear Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicción de los resultados con el conjunto de testing
Y_pred = classifier.predict(X_test)

# Elaborar una matriz  de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)