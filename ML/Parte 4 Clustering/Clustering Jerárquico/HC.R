# Clustering Jerárquico

# Importar los datos
dataset = read.csv("Mall_Customers.csv")
X = dataset[, 4:5]

# Utilizar el dendograma para encontrar el número óptimo de clusters
dendogram = hclust(dist(X, method ="euclidean"), 
                   method = "ward.D")
plot(dendogram,
     main = "Dendograma",
     xlab = "Clientes del Centro Comercial",
     ylab = "Distancia Euclídea")

# Ajustar el clustering jerárquico a nuestro dataset
hc = hclust(dist(X, method ="euclidean"), 
                   method = "ward.D")

y_hc = cutree(hc, k = 5)

# Visualizar los clusters
#install.packages("cluster")
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion (1-100)")