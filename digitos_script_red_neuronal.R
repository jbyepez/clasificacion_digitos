# Cargar datos de entrenamiento
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)

# Cargar datos de prueba
testingData <- read.csv("./data/mnist_test.csv",header = FALSE)

library(keras)

# Labels (columna 1)
trainY <- trainingData["V1"] %>% as.matrix() %>% as.numeric()
trainLabels <- to_categorical(trainY)
testY <- testingData["V1"] %>% as.matrix() %>% as.numeric()
testLabels <- to_categorical(testY)

# Transformar los datos, para el modelo 1 (acepta vectores)
trainingData <- trainingData %>%
  subset(select = -c(V1)) %>%
  as.matrix() %>%
  as.numeric() %>%
  array(c(60000,784))

testingData <- testingData %>%
  subset(select = -c(V1)) %>%
  as.matrix() %>%
  as.numeric() %>%
  array(c(10000,784))

# Transformar los datos para el modelo 2 (acepta matrices)
trainingData2D <- array(trainingData, c(60000,28,28,1))

testingData2D <- array(testingData, c(10000,28,28,1))

#Ver uno de los digitos como imagen
library(magick)
im <- function(x,m,n){ #Funcion para ver el digito
  plot(image_read(aperm(array(m[x,],c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  return(n[x]) #Muestra el numero respectivo en consola
}
#Imagen 4 de trainingData
im(4,trainingData,trainY)
#Imagen 8 de testingData
im(8,testingData,testY)

# Primer modelo, tomado de: https://nextjournal.com/gkoehler/digit-recognition-with-keras
model1 <- keras_model_sequential() %>%
  layer_dense(512, input_shape=c (784,NULL), activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(10, activation = 'softmax') %>%
  compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

# Entrenar primer modelo
history1 <- model1 %>%
  fit(trainingData,
      trainLabels,
      epochs = 100,
      batch_size = 2048,
      validation_split = 0.2)
#validation_data = list(test, testLabels))
plot(history)

# Segundo modelo
model2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(5,5), input_shape = c(28,28,1), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size=c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size=c(2, 2)) %>%
  layer_dropout(0.2) %>%
  layer_flatten() %>%
  layer_dense(128, activation='relu') %>%
  layer_dense(10, activation='softmax') %>%
  compile(loss='categorical_crossentropy', optimizer='adam', metrics=c('accuracy'))
# Modelo tomado de: https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8

# Entrenar modelo
history2 <- model2 %>% 
  fit(trainingData2D,
      trainLabels,
      validation_split = 0.2,
      epochs=10,
      batch_size=200)


#Guardar
save_model_hdf5(model1, "model1")
save_model_hdf5(model2, "model2")

#Cargar
model1 <- load_model_hdf5("model1")
model2 <- load_model_hdf5("model2")

# Evaluation & Prediction - train data
model1 %>% evaluate(trainingData, trainLabels)
predtrain <- model1 %>% predict_classes(trainingData)
a <- table(Predicted = predtrain, Actual = trainY)

prob <- model1 %>% predict_proba(trainingData)
cbind(prob, Predicted_class = predtrain, Actual = trainY)

# Conjunto de imagenes de entrenamiento mal predichas
badPredTrainData <- cbind(predicted = predtrain, correct = trainY, pos = c(1:60000))
badPredTrainData <- subset(badPredTrainData, badPredTrainData[,1] != badPredTrainData[,2])

# Evaluation & Prediction - test data
model1 %>% evaluate(testingData, testLabels)
predtest <- model1 %>% predict_classes(testingData)
b <- table(Predicted = predtest, Actual = testY)

prob <- model1 %>% predict_proba(testingData)
cbind(prob, Predicted_class = predtest, Actual = testY)

# Conjunto de imagenes de prueba mal predichas
badPredTestData <- cbind(predicted = predtest, correct = testY, pos = c(1:10000))
badPredTestData <- subset(badPredTestData, badPredTestData[,1] != badPredTestData[,2])

#Funcion para ver uno de los digitos mal predichos
library(magick)
badpred <- function(predictions,dataset,pos){ #Funcion para ver el digito
  plot(image_read(aperm(array(as.numeric(dataset[predictions[pos,3],]),c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
  print(paste("predicho:",predictions[pos,1]))
  print(paste("real:    ",predictions[pos,2]))
}

#Ejemplo de uso: Imagen mal predicha 4 de los datos de prueba
badpred(badPredTestData,testingData,4)
