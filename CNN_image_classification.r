library(keras)

# Load the zip.train data file from https://web.stanford.edu/~hastie/ElemStatLearn/data.html - we will 
# be splitting this data set into train/test so please ignore the zip.test data set (which is a 
# different file). For this data set no need to scale the inputs, because they are already in [-1,1]
if(!file.exists("zip.train.gz")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz",
    "zip.train.gz")
}
zip.train <- data.table::fread("zip.train.gz")

zip.X.array <- array(
  unlist(zip.train[1:nrow(zip.train),-1]),
  c(nrow(zip.train), 16, 16, 1))


zip.class.tab <- table(zip.train$V1)
num_classes <- length(zip.class.tab)
zip.y.mat <- keras::to_categorical(zip.train$V1,num_classes)
batch_size <- 128
epochs <- 20


set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(zip.train)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test
x_train <- zip.X.array[is.train, , ,]
y_train <- zip.y.mat[is.train, ]
x_test <- zip.X.array[is.test, , , ] 
y_test <- zip.y.mat[is.test, ]

test <- array(x_test, c(1459, 16, 16, 1))


##conv.units <- c(784,6272,9216,128,10)
##dense.units <- c(784,270,270,128,10)

  
## convolution
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = dim(zip.X.array)[-1]) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
  )
  
result <- model %>% fit(
  test, y_test,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
  )

some <- result$val_loss

scores <- model %>% evaluate(
  test, y_test, verbose = 0
)

metrics.wide <- do.call(data.table::data.table, result$metrics)
metrics.wide[, epoch := 1:.N]  
str(metrics.wide)
