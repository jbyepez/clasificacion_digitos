#Cargar datos de entrenamiento
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)

#Cargar datos de prueba
testingData <- read.csv("./data/mnist_test.csv",header = FALSE)



library(keras)

#Labels
trainY <- as.matrix(trainingData["V1"])
trainLabels <-to_categorical(trainY)
testY <- as.matrix(testingData["V1"])
testLabels <- to_categorical(testY)

trainingData <- array(as.matrix(subset(trainingData, select = -c(V1))), c(60000,784))
testingData <- array(as.matrix(subset(testingData, select = -c(V1))), c(10000,784))

# #Ver uno de los digitos como imagen
# library(magick)
# im <- function(x,m,n){ #Funcion para ver el digito
#   plot(image_read(aperm(array(as.numeric(m[x,,1]),c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
#   return(n[x,1]) #Muestra el numero respectivo en consola
# }
# #Imagen 4 de trainingData
# im(4,trainingData,trainY)
# #Imagen 8 de testingData
# im(8,testingData,testY)


# Crear modelo
model <- keras_model_sequential() %>%
  layer_dense(128, input_shape=c (784,NULL), activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(10, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(10, activation = 'softmax') %>%
  compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')


# Entrenar modelo
history2 <- model %>%
  fit(trainingData,
      trainLabels,
      epochs = 100,
      batch_size = 2048,
      validation_split = 0.2)
      #validation_data = list(test, testLabels))
plot(history)

#Guardar
save_model_hdf5(model, "model")

#Cargar
model <- load_model_hdf5("model")

# Evaluation & Prediction - train data
model %>% evaluate(trainingData, trainLabels)
predtrain <- model %>% predict_classes(trainingData)
a <- table(Predicted = predtrain, Actual = trainY)

prob <- model %>% predict_proba(trainingData)
cbind(prob, Predicted_class = predtrain, Actual = trainY)

# Conjunto de imagenes de entrenamiento mal predichas
badPredTrainData <- cbind(predicted = predtrain, correct = trainY, pos = c(1:60000))
badPredTrainData <- subset(badPredTrainData, badPredTrainData[,1] != badPredTrainData[,2])

# Evaluation & Prediction - test data
model %>% evaluate(testingData, testLabels)
predtest <- model %>% predict_classes(testingData)
b <- table(Predicted = predtest, Actual = testY)

prob <- model %>% predict_proba(testingData)
cbind(prob, Predicted_class = predtest, Actual = testY)

# Conjunto de imagenes de prueba mal predichas
badPredTestData <- cbind(predicted = predtest, correct = testY, pos = c(1:10000))
badPredTestData <- subset(badPredTestData, badPredTestData[,1] != badPredTestData[,2])

#Funcion para ver uno de los digitos mal predichos
library(magick)
badpred <- function(predictions,dataset,pos){ #Funcion para ver el digito
  plot(image_read(aperm(array(as.numeric(dataset[predictions[pos,3],]),c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  print(paste("predicho:",predictions[pos,1]))
  paste("real:    ",predictions[pos,2])
  #return(n[x,1]) #Muestra el numero respectivo en consola
}

#Ejemplo de uso: Imagen mal predicha 4 de los datos de prueba
badpred(badPredTestData,testingData,4)
