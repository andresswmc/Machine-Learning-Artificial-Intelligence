# Natural Language Processing

# Importar el dataset

dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '', 
                     stringsAsFactors = FALSE)

# Limpieza de texto
# install.packages("tm")
# install.packages("SnowballC")

library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))

# Consultar el primer elemento del corpus
# as.character(corpus[[1]])

corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Crear el modelo Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


dataset$Liked = factor(dataset$Liked, levels = c(0,1))

library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Ajustar el Random Forest con el conjunto de entrada
#install.packages("randomForest")
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicción de los resultados con el conjunto de testing
Y_pred = predict(classifier, newdata = testing_set[, -692])

# Crear la matriz de confusión
cm = table(testing_set[, 692], Y_pred)