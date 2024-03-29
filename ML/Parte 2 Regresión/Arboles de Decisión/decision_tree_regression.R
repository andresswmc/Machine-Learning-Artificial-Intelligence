# �rbol de Decisi�n de regresi�n

# Plantilla de Regresi�n

# Importar el dataset

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)

# Ajustar Modelo de Regresi�n con el conjunto de Datos
#install.packages("rpart")
library(rpart)
regression = rpart(formula = Salary ~ .,
                   data = dataset,
                   control = rpart.control(minsplit = 1))


# Predicci�n de nuevos resultados con Regresi�n 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualizaci�n del �rbol de regresi�n

# install.packages("ggplot2")
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Predicci�n (Modelo de Regresi�n)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")
