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
im(4,trainingData)
#Imagen 8 de testingData
im(8,testingData)

