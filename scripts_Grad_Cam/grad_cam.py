from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import time
from src.utils import plot_loss_curves_mae, build_dataset,normalize_img
import numpy as np

# repertoire des donnees
path_dir = Path(r'F:\Projet_Datascientest')
output_dir ="./remplissage"
IMG_PATH = path_dir/'Remplissage'

mode = 'inference'

#mode = 'train'


def show_images_from_ds(ds, class_names=None, channels=3):
  """
  Args:
    ds (tf_Dataset): dataset
  """
  data = next(iter(ds))
  cmap_color = 'viridis'
  if channels == 1:
    cmap_color = 'gray'
  plt.rc('font', size=10)

  for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(data[0][i], cmap=cmap_color)  # images are already resized and scaled
    if (class_names is not None):
      plt.title(class_names[data[1][i]] + str((tf.shape(data[0][i])).numpy()))
    else:
      plt.title("image size : " + str((tf.shape(data[0][i])).numpy()))
    plt.axis(False)
  plt.show()

# recuperation des fichiers
###################################################################################
input_img_paths = sorted(
    [
        os.path.join(IMG_PATH, fname)
        for fname in os.listdir(IMG_PATH)
        if (fname.endswith(".jpeg") or fname.endswith(".png"))
    ]
)

df = pd.DataFrame(input_img_paths, columns=['image'])
df['file'] = [Path(filepath).stem for filepath in df['image']]
df['extension'] = [Path(filepath).suffix for filepath in df['image']]
df['taux'] = [int(filename.split("_")[-1])/100. for filename in df['file']]
print(df.tail())
print(len(df[df['extension']=='.png']))

X_temp, X_test, y_temp, y_test = train_test_split(df['image'],df['taux'], test_size=0.2, random_state=1234)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=1234)
print("y_train",len(y_train))
print("y_val",len(y_val))
print("y_test",len(y_test))
#
#
# filename = r'F:\Projet_Datascientest\Remplissage\c01583_40.png'
# print(filename)
# image_string = tf.io.read_file(filename)
# image = tf.image.decode_png(image_string, channels=3)
# print(tf.reduce_max(image))
#
# build dataset

img_size = 128
batch_size = 16
##################################   TRAIN    #################################################
if (mode=='train'):

  ds_train = build_dataset(X_train, y_train,
                    True,
                    is_augm=True,
                    scale=False,            # -_-' no scaling for EfficientNet from keras.applications!
                    img_size = img_size,
                    data_from = "files",   # files, tfrecords, (pixels) values
                    batch_size = batch_size,
                    drop_remainder=False)
  #
  ds_val = build_dataset(X_val, y_val,
                    False,
                    is_augm=False,
                    scale=False,            # -_-' no scaling for EfficientNet from keras.applications!
                    img_size = img_size,
                    data_from = "files",   # files, tfrecords, (pixels) values
                    batch_size = batch_size,
                    drop_remainder=False)

  ds_test = build_dataset(X_test, y_test,
                    False,
                    is_augm=False,
                    scale=False,            # -_-' no scaling for EfficientNet from keras.applications!
                    img_size = img_size,
                    data_from = "files",   # files, tfrecords, (pixels) values
                    batch_size = 32,
                    drop_remainder=False)

  #show_images_from_ds(ds_train)

  # build model
  ###################################################################################

  # model = Sequential()
  # #model.add(base_model)
  # #model.add(GlobalAveragePooling2D())
  # model.add(Input(shape=(img_size,img_size,3)))
  # model.add(Dense(2048, activation="relu"))
  # model.add(Dense(1024,activation="relu"))
  # model.add(Dense(512,activation="relu"))
  # model.add(Dense(256, activation="relu"))
  # model.add(Dense(128, activation="relu"))
  # model.add(Dense(1))
  # model.summary()

  def create_cnn(width, height, depth, filters=(16, 32, 64, 64, 128), regress=True):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
      # if this is the first CONV layer then set the input
      # appropriately
      if i == 0:
        x = inputs
      # CONV => RELU => BN => POOL
      x = Conv2D(f, (3, 3), padding="same")(x)
      x = Activation("relu")(x)
      if (i==0 or i==1):
          x = BatchNormalization(axis=chanDim)(x)
          x = MaxPooling2D(pool_size=(2, 2))(x)
      # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(128)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
      x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

  def efficient_cnn(width, height, depth):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)

    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    unfreeze_layers = 13
    base_model = EfficientNetB0(include_top=False, input_shape=(img_size, img_size, 3))

    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
      layer.trainable = False

    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
###
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
###
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(128)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

  model = efficient_cnn(img_size, img_size, 3)
  #model = create_cnn(img_size, img_size, 3, regress=True)
  #opt = Adam(lr=1e-2, decay=1e-3 / 20)

  def scheduler(epoch, lr):
    if epoch < 5:
      return lr
    else:
      return lr * tf.math.exp(-0.1)
  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  callback_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

  opt = Adam(lr=1e-3)
  #model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
  model.compile(loss="mse", optimizer=opt, metrics=['mae'])
  model.summary()
  # train the model
  print("[INFO] training model...")
  history = model.fit(ds_train,
            validation_data=ds_val,
            epochs=200,
            callbacks=[callback,callback_stop])

  plt.figure(figsize=(8, 8))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.show()

  model.save('model_remplissage.h5')

##################################   TEST    #################################################
else:
  ds_test = build_dataset(X_test, y_test,
                    False,
                    is_augm=False,
                    scale=False,            # -_-' no scaling for EfficientNet from keras.applications!
                    img_size = img_size,
                    data_from = "files",   # files, tfrecords, (pixels) values
                    batch_size = 32,
                    drop_remainder=False)

  model = tf.keras.models.load_model('model_remplissage.h5')
  base_model = EfficientNetB0(include_top=False, input_shape=(img_size, img_size, 3))
  base_model.summary()
  y_pred=model.predict(ds_test)


  def plot_preds(y_test,y_pred):
    t1 = np.arange(0., len(y_test))
    t2 = np.arange(0., len(y_pred))
    plt.figure(figsize=(12,6))
    plt.scatter(t1,y_test, c="b", label="True values")
    plt.scatter(t2,y_pred, c="g", label="Predictions")
    plt.legend()
    plt.show()


  # plot_preds(y_test, y_pred)
  for i in range(len(y_test)):
      print(y_test.iloc[i], y_pred[i])


  def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


  print(model.layers[1].layers[-3].name)
  def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
      #[model.inputs], [model.layers[1].get_layer(last_conv_layer_name).output, model.output]
      [model.inputs], [model.layers[1].layers[-3].output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
      last_conv_layer_output, preds = grad_model(img_array)
      # if pred_index is None:
      #   pred_index = tf.argmax(preds[0])
      # class_channel = preds[:, pred_index]
      loss = preds[0]
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


  # Prepare image
  input_image = r'F:\Deep_Learning\SEGA\datasets\Remplissage\c00404_25.jpeg'
  preprocess_input = tf.keras.applications.efficientnet.preprocess_input
  img_array = preprocess_input(get_img_array(input_image, (img_size,img_size)))
  last_conv_layer_name = "top_conv"

  # Print what the top predicted class is
  preds = model.predict(img_array)
  print("Predicted:", preds)

  # Generate class activation heatmap
  heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

  # Display heatmap
  plt.matshow(heatmap)
  plt.show()



