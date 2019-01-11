# Image classification with keras and explain models with lime

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
####


###

### Explaining Keras image classification models with lime ###
#https://shirinsplayground.netlify.com/2018/06/keras_fruits_lime/
# Thomas wrote a very nice article about how to use keras and lime in R! Here, I am following this article to use Imagenet (VGG16) to make and explain predictions of fruit images and then I am extending the analysis to last week’s model and compare it with the pretrained net.
# install.packages("keras")
# install.packages("lime")
# install.packages("magick")
library(keras)   # for working with neural nets
library(lime)    # for explaining models
library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting
library(reticulate)
# use_python('/home/upsman/anaconda3/envs/env-python37/bin/python')
# use_python('/opt/anaconda/anaconda3/envs/r-tensorflow/bin/python')
# use_python("/home/upsman/anaconda3/lib/python3.7/site-packages/keras")
# py_config()
#all above don't work:

# install_keras(method = c("auto", "virtualenv", "conda"), conda = "auto", tensorflow = "default", extra_packages = NULL)

#test
# library(keras)
# use_condaenv("r-tensorflow",required=T)
# data <- dataset_mnist()
# library(reticulate)
# py_module_available('keras') # must return TRUE
# py_module_available('tensorflow') # must return TRUE
# py_discover_config("keras") # more info on the python env, tf and keras
# py_config()
# 
# library(tensorflow)

##
# Loading the pretrained Imagenet model
model <- application_vgg16(weights = "imagenet", include_top = TRUE)
model

# loading my own model from last week’s post
model2 <- load_model_hdf5(filepath = "/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/keras/fruits_checkpoints.h5")
model2

# Load and prepare images #
# Here, I am loading and preprocessing two images of fruits (and yes, I am cheating a bit because I am choosing images where I expect my model to work as they are similar to the training images…).

# Banana
test_image_files_path <- "/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/Test"

img <- image_read('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/272px-Banana-Single.jpg')
img_path <- file.path(test_image_files_path, "Banana", 'banana.jpg')
image_write(img, img_path)
#plot(as.raster(img))
Clementine
img2 <- image_read('https://cdn.pixabay.com/photo/2010/12/13/09/51/clementine-1792_1280.jpg')
img_path2 <- file.path(test_image_files_path, "Clementine", 'clementine.jpg')
image_write(img2, img_path2)
#plot(as.raster(img2))


# Superpixels #

plot_superpixels(img_path, n_superpixels = 35, weight = 10)
plot_superpixels(img_path2, n_superpixels = 50, weight = 20)
#From the superpixel plots we can see that the clementine image has a higher resolution than the banana image.

# Prepare images for Imagenet #
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
# test predictions
res <- predict(model, image_prep(c(img_path, img_path2)))
imagenet_decode_predictions(res)

# load labels and train explainer
model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
explainer <- lime(c(img_path, img_path2), as_classifier(model, model_labels), image_prep)
# Training the explainer (explain() function) can take pretty long. It will be much faster with the smaller images in my own model but with the bigger Imagenet it takes a few minutes to run.

explanation <- explain(c(img_path, img_path2), explainer, 
                       n_labels = 2, n_features = 35,
                       n_superpixels = 35, weight = 10,
                       background = "white")
# plot_image_explanation() only supports showing one case at a time
plot_image_explanation(explanation)

clementine <- explanation[explanation$case == "clementine.jpg",]
plot_image_explanation(clementine)

# Prepare images for my own model
# test predictions (analogous to training and validation images)
test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_image_files_path,
  test_datagen,
  target_size = c(20, 20),
  class_mode = 'categorical')

predictions <- as.data.frame(predict_generator(model2, test_generator, steps = 1))

load("/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/fruits_classes_indices.RData")
fruits_classes_indices_df <- data.frame(indices = unlist(fruits_classes_indices))
fruits_classes_indices_df <- fruits_classes_indices_df[order(fruits_classes_indices_df$indices), , drop = FALSE]
colnames(predictions) <- rownames(fruits_classes_indices_df)

t(round(predictions, digits = 2))

for (i in 1:nrow(predictions)) {
  cat(i, ":")
  print(unlist(which.max(predictions[i, ])))
}

# This seems to be incompatible with lime, though (or if someone knows how it works, please let me know) - so I prepared the images similarly to the Imagenet images.

image_prep2 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(20, 20))
    x <- image_to_array(img)
    x <- reticulate::array_reshape(x, c(1, dim(x)))
    x <- x / 255
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
# prepare labels
fruits_classes_indices_l <- rownames(fruits_classes_indices_df)
names(fruits_classes_indices_l) <- unlist(fruits_classes_indices)
fruits_classes_indices_l

# train explainer
explainer2 <- lime(c(img_path, img_path2), as_classifier(model2, fruits_classes_indices_l), image_prep2)
explanation2 <- explain(c(img_path, img_path2), explainer2, 
                        n_labels = 1, n_features = 20,
                        n_superpixels = 35, weight = 10,
                        background = "white")
# plot feature weights to find a good threshold for plotting block (see below)
explanation2 %>%
  ggplot(aes(x = feature_weight)) +
  facet_wrap(~ case, scales = "free") +
  geom_density()


# plot predictions
plot_image_explanation(explanation2, display = 'block', threshold = 5e-07)

clementine2 <- explanation2[explanation2$case == "clementine.jpg",]
plot_image_explanation(clementine2, display = 'block', threshold = 0.16)

sessionInfo()



