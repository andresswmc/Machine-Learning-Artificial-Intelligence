# Plantilla para el Pre Procesado de Datos - Datos Categóricos
# Importar el dataset
dataset = read.csv('Data.csv', stringsAsFactors = F)

str(dataset)
# Codificar las variables categóricas
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0,1))
str(dataset)

# Dividir los datos en conjunto de entrenamiento y Test

#install.packages("caTools")

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)