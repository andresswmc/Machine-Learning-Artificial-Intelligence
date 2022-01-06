# Random Forest Regression

# Plantilla de Regresión

# Importar el dataset

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)

# Ajustar Modelo Random Forest con el conjunto de Datos
#install.packages("randomForest")
library(randomForest)
set.seed(1234)
regression = randomForest(x = dataset[1],
                          y = dataset$Salary,
                          ntree = 500)



# Predicción de nuevos resultados con Random Forest 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualización del modelo Random Forest

# install.packages("ggplot2")
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Predicción (Random Forest)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")