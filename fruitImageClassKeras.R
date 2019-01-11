# Image classification with keras in roughly 100 lines of code.

#https://shirinsplayground.netlify.com/2018/06/keras_fruits/
#https://www.kaggle.com/moltean/fruits/data
# install.packages("keras")
library(keras)
# list of fruits to modle
fruit_list <- c("Kiwi", "Banana", "Apricot", "Avocado", "Cocos", "Clementine", "Mandarine", "Orange",
                "Limes", "Lemon", "Peach", "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate")

# number of output classes (i.e. fruits)
output_n <- length(fruit_list)

# image size to scale down to (original images are 100 x 100 px)
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "fruits/fruits-360/Training/"
valid_image_files_path <- "fruits/fruits-360/Test/"


### Loading images ###

# optional data augmentation
train_data_gen = image_data_generator(
  rescale = 1/255 #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
)

# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
  )  

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                          train_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = fruit_list,
                                          seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                          valid_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = fruit_list,
                                          seed = 42)

cat("Number of images per class:")

## Number of images per class:

table(factor(train_image_array_gen$classes))

cat("\nClass label vs index mapping:\n")

## 
## Class label vs index mapping:

train_image_array_gen$class_indices

fruits_classes_indices <- train_image_array_gen$class_indices
save(fruits_classes_indices, file = "fruits_classes_indices.RData")



### Define model ###

# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10


# The model I am using here is a very simple sequential convolutional neural net with the following hidden layers: 2 convolutional layers, one pooling layer and one dense layer.

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%

  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%

  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Fit the model; because I used image_data_generator() and flow_images_from_directory() I am now also using the fit_generator() to run the training.

# fit
hist <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("fruits/fruits-360/keras/fruits_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "fruits/fruits-360/keras/logs")
  )
)

# In RStudio we are seeing the output as an interactive plot in the “Viewer” pane but we can also plot it:

plot(hist)


# As we can see, the model is quite accurate on the validation data. However, we need to keep in mind that our images are very uniform, they all have the same white background and show the fruits centered and without anything else in the images. Thus, our model will not work with images that don’t look similar as the ones we trained on (that’s also why we can achieve such good results with such a small neural net).

# Finally, I want to have a look at the TensorFlow graph with TensorBoard.

tensorboard("fruits/fruits-360/keras/logs")

# you could now save your model and/or the weights, visualize the hidden layers, run predictions on test data, etc. 
sessionInfo()
