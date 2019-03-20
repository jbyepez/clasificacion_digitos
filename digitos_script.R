#load("clase_svm/df17.RData")
#load("clase_svm/df56.RData")

#Cargar datos de entrenamiento
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)

#Cargar datos de prueba
testingData <- read.csv("./data/mnist_test.csv",header = FALSE)


#Ver uno de los digitos como imagen
library(magick)
im <- function(x,m){ #Funcion para ver el digito
  plot(image_read(aperm(array(as.numeric(m[x,2:785]),c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  return(m[x,1]) #Muestra el numero respectivo en consola
}

#Imagen 4 de trainingData
im(3,trainingData)

#Imagen 8 de testingData
im(8,testingData)



#********************Descriptiva**************************#

# cantidad de datos por digito 
barplot(table(trainingData$V1))



#****************** Preparando datos train (muestra) ***************#
# transformemos las vriables a numericas
trainingData <- apply(trainingData, 2, as.numeric)
trainingData <- as.data.frame(trainingData)
trainingData$V1 <- as.factor(trainingData$V1)
apply(trainingData, 2, sum)


# tomemos una muestra por que mi pc no aguanta
muestra <- sample(1:6000, 1000)
trainingData2 <- as.data.frame(trainingData[muestra,])



#****************** Preparando datos test (muestra) ***************#
# transformemos las vriables a numericas
testingData <- apply(testingData, 2, as.numeric)
testingData <- as.data.frame(testingData)
testingData$V1 <- as.factor(testingData$V1)





# *************** Modelos ***********************#
# 1) regresion logistica multinomial (hay problemas con este modelo)

# transformemos la variable respuesta
library(nnet)
trainingData$V1 <- factor(trainingData$V1)
modelo <- multinom(V1~V2+V3, trainingData2)
summary(modelo)
apply(trainingData2, 2, function(x) sum(is.na(x)))


library ( MASS )
fit <-  polr (V1~V2, data = trainingData2)
summary ( fit )




#2) Arboles de clasificacion**
library(rpart)
clasificadorDT <- rpart(V1 ~ ., data = trainingData2)
# method = "anova", "poisson", "class" or "exp"
summary(clasificadorDT)
pred_valid_DT <- predict(clasificadorDT, newdata = testingData, type = 'class')
matrizConfusion <- table(testingData$V1, pred_valid_DT)
matrizConfusion
clasificacion_correctas <- sum(diag(matrizConfusion)) / sum(matrizConfusion)
# 0.61


#3) Bosques aleatorio
library(randomForest)
clasificadorRF <- randomForest(V1 ~ ., data = trainingData2, ntree = 250)
pred_valid_RF <- predict(clasificadorRF, newdata = testingData)
matrizConfusion <- table(testingData$V1, pred_valid_RF)
matrizConfusion
clasificacion_correctas <- sum(diag(matrizConfusion)) / sum(matrizConfusion)
#0.90




# 4) Máquinas de soporte vectorial 
library(e1071)
clasificadorSVM <- svm(V1 ~ ., data = trainingData2, 
                       type = 'C-classification', kernel = 'linear') #probar polynomial radial y sigmoid
pred_valid_svm <- predict(clasificadorSVM, newdata = testingData)
matrizConfusion <- table(testingData$V1, pred_valid_svm)
matrizConfusion
clasificacion_correctas <- sum(diag(matrizConfusion)) / sum(matrizConfusion)


# 5) Redes neuronales
library(neuralnet)
ann <- neuralnet(V1 ~ V2, trainingData2, hidden = 10, rep = 3)


