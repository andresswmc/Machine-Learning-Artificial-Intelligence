# Clustering con K-means

# Importar los datos
dataset = read.csv("Mall_Customers.csv")
X = dataset[, 4:5]

# M�todo del codo
set.seed(6)
wcss = vector()
for (i in 1:10){
  wcss[i] <- sum(kmeans(X, i)$withinss)
}
plot(1:10, wcss, type = 'b', main = " M�todo del codo",
     xlab = "N�mero de clusters (k)", ylab = "WCSS(k)")

# Aplicar el algoritmo de k-means con k �ptimo
set.seed(29)
kmeans <-  kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualizaci�n de los clusters
#install.packages("cluster")
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion (1-100)")
