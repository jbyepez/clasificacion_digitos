#Cargar datos de entrenamiento
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)

#Cargar datos de prueba
testingData <- read.csv("./data/mnist_test.csv",header = FALSE)


# #Ver uno de los digitos como imagen
# library(magick)
# im <- function(x,m){ #Funcion para ver el digito
#   plot(image_read(aperm(array(as.numeric(m[x,2:785]),c(28,28,1)),c(2,1,3)))) #Muestra la imagen en el viewer
#   return(m[x,1]) #Muestra el numero respectivo en consola
# }
# #Imagen 4 de trainingData
# im(4,trainingData)
# #Imagen 8 de testingData
# im(8,testingData)
library(keras)

#Labels
trainY <- as.matrix(trainingData["V1"])
trainLabels <-to_categorical(trainY)
testY <- as.matrix(testingData["V1"])
testLabels <- to_categorical(testY)

trainingData <- subset(trainingData, select = -c(V1))
trainingData <- as.matrix(trainingData)
trainingData <- array(trainingData, c(60000,784,1))
testingData <- subset(testingData, select = -c(V1))

model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 8, kernel_size = 3, activation = 'relu', input_shape = c(784, 1)) %>% 
  layer_conv_1d(filters = 8, kernel_size = 3, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 16, kernel_size = 3, activation = 'relu') %>% 
  layer_conv_1d(filters = 16, kernel_size = 3, activation = 'relu') %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'sigmoid') %>% 
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'rmsprop',
    metrics = c('accuracy')
  )

# Fit model
history <- model %>%
  fit(trainingData,
      trainLabels,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2)
      #validation_data = list(test, testLabels))
plot(history)