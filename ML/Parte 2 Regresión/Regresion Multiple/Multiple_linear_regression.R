# Importar el dataset

dataset = read.csv("50_Startups.csv")

#codificar las variables categóricas
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1, 2, 3))

# Dividir los datos en los conjuntos de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de Regresión Lineal Múltiple con el conjunto de Entrenamiento
regression = lm(formula = Profit ~ .,
                data = training_set)

# Predecir los resultados con el conjunto de testing

Y_pred = predict(regression, newdata = testing_set)

# Construir un modelo Óptimo con la Eliminación hacia atrás
SL = 0.05

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,                  ,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,                  ,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regression)

# Regresión Lineal Múltiple en R - Eliminación hacia atrás automática

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
