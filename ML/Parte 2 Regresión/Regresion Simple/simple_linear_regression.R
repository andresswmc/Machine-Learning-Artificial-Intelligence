dataset = read.csv("Salary_Data.csv")

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de regresión lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predecir resultados con el conjunto de test
Y_pred = predict(regressor, newdata = testing_set)

#Visualizacion de los resultados en el conjunto de entrenamiento
#install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
             colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)") +
  xlab("Años de experiencia") +
  ylab("Sueldo (en $)")

#Visualizacion de los resultados en el conjunto de test

ggplot() +
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de test)") +
  xlab("Años de experiencia") +
  ylab("Sueldo (en $)")