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
epochs <- 30


set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(zip.train)))
test.fold <- 1
is.test <- fold.vec == test.fold
table( is.train )
is.train <- !is.test
x_train <- array(zip.X.array[is.train, , ,] , c(5832 , 16, 16 ,1 ))
y_train <- zip.y.mat[is.train, ]
x_test <- array(zip.X.array[is.test, , , ] , c( 1459, 16 , 16 , 1))
y_test <- zip.y.mat[is.test, ]

test <- array(x_test, c(1459, 16, 16, 1))


##conv.units <- c(256,6272,9216,128,10)
##dense.units <- c(256,270,270,128,10)

  
## convolution
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = dim(zip.X.array)[-1]) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = num_classes, activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
  )
  
conv_result <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
  )

conv_metrics.wide <- do.call(data.table::data.table, conv_result$metrics)
conv_metrics.wide[, epoch := 1:.N]  
str(conv_metrics.wide)

conv_best_epoch = which.min(conv_metrics.wide$val_loss)


## dense

model <- keras_model_sequential() %>%
  layer_flatten( input_shape = dim(zip.X.array)[-1]) %>%                                 # input layer
  layer_dense(units = 270 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
  layer_dense(units = 270 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
  layer_dense(units = 128 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
  layer_dense(units= 10 , activation = "softmax", use_bias = FALSE )         # ouput layer

model %>% 
  compile(
    loss = loss_categorical_crossentropy,
    optimizer = "SGD",
    metrics = "accuracy"
  )

dense_result <- model %>% 
  fit(
    x = x_train, y = y_train,
    epochs = epochs,
    validation_split = 0.2,
    verbose = 2
  )

dense_metrics.wide <- do.call(data.table::data.table, dense_result$metrics)
dense_metrics.wide[, epoch := 1:.N]  
str(dense_metrics.wide)

dense_best_epoch = which.min(dense_metrics.wide$val_loss)



### Refitting according to best epochs

## Convolutional
conv_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = dim(zip.X.array)[-1]) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = num_classes, activation = 'softmax')

conv_model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

conv_result <- conv_model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = conv_best_epoch,
  validation_split = 0
)

conv_test_metrics.wide <- do.call(data.table::data.table, conv_result$metrics)
conv_test_metrics.wide[, epoch := 1:.N]  
str(conv_test_metrics.wide)

## Dense

dense_model <- keras_model_sequential() %>%
  layer_flatten( input_shape = dim(zip.X.array)[-1]) %>%                                 # input layer
  layer_dense(units = 270 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
  layer_dense(units = 270 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
  layer_dense(units = 128 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
  layer_dense(units= 10 , activation = "softmax", use_bias = FALSE )         # ouput layer

dense_model %>% 
  compile(
    loss = loss_categorical_crossentropy,
    optimizer = "SGD",
    metrics = "accuracy"
  )

dense_result <- dense_model %>% 
  fit(
    x = x_train, y = y_train,
    epochs = dense_best_epoch,
    validation_split = 0,
    verbose = 2
  )

dense_test_metrics.wide <- do.call(data.table::data.table, dense_result$metrics)
dense_test_metrics.wide[, epoch := 1:.N]  
str(dense_test_metrics.wide)


### Predict on Test data

test_class <- apply(y_test, 1, which.max)

conv_pred <- predict( conv_model, x_test)
conv_pred_class <- apply(conv_pred, 1, which.max)
conv_accuracy.vec <- conv_pred_class == test_class
conv_accuracy <- sum(conv_accuracy.vec)

dense_pred <- predict( dense_model , x_test)
dense_pred_class <- apply(dense_pred, 1, which.max)
dense_accuracy.vec <- dense_pred_class == test_class
dense_accuracy <- sum(dense_accuracy.vec)

# baseline accuracy
baseline <- sum(1 == test_class)  ## which.max(zip.class.tab)






