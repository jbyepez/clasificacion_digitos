
trainingData <- read.csv("./data/mnist_train.csv",header = FALSE)
dm <- as.matrix.data.frame(trainingData)
im <- function(x){
  plot(image_read(aperm(array(as.numeric(dm[x,2:785]),c(28,28,1)),c(2,1,3))))
  print(dm[x,1])
}
