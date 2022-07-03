import logging
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
from PIL import Image
import tensorflow as tf

def set_logger(log_path):
  """Sets the logger to log info in terminal and file `log_path`.

  In general, it is useful to have a logger so that every output to the terminal is saved
  in a permanent file. Here we save it to `model_dir/train.log`.

  Example:
  ```
  logging.info("Starting training...")
  ```

  Args:
      log_path: (string) where to log
  """
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

def display_top_masks(image, mask, class_ids, bbox, booldisp, output_dir, limit=1):
  """Display the given image and the top few class masks."""
  to_display = []
  titles = []
  titles.append("Original image H x W={}x{}".format(image.shape[0], image.shape[1]))
  to_display.append(image)
  titles.append("Bounding box")
  image_bbox = image.copy()
  for i in range(bbox.shape[0]):
    x, y, w, h = bbox[i, :]
    cv2.rectangle(image_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
  to_display.append(image_bbox)

  # Pick top prominent classes in this image
  unique_class_ids = np.unique(class_ids)
  mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
               for i in unique_class_ids]
  top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                  key=lambda r: r[1], reverse=True) if v[1] > 0]
  # Generate images and titles
  for i in range(limit):
    # class_id = top_ids[i] if i < len(top_ids) else -1
    # Pull masks of instances belonging to the same class.
    # m = mask[:, :, np.where(class_ids == class_id)[0]]
    m = mask[:, :, :]
    m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
    to_display.append(m)
    # titles.append(class_names[class_id] if class_id != -1 else "-")
    titles.append("Masks of segmentation")
  if booldisp:
    filename = "display_mask_" + str(booldisp) + ".png"
    display_images(to_display, titles=titles, cols=limit + 1,
                   cmap="Blues_r", output_dir=output_dir, filename=filename)
  return to_display

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None, output_dir=".", filename="display_mask.png"):
  """Display the given set of images, optionally with titles.
  images: list or array of image tensors in HWC format.
  titles: optional. A list of titles to display with each image.
  cols: number of images per row
  cmap: Optional. Color map to use. For example, "Blues".
  norm: Optional. A Normalize instance to map values to colors.
  interpolation: Optional. Image interpolation to use for display.
  """
  titles = titles if titles is not None else [""] * len(images)
  rows = len(images) // cols + 1
  plt.figure(figsize=(14, 14 * rows // cols))
  i = 1
  for image, title in zip(images, titles):
    plt.subplot(rows, cols, i)
    plt.title(title, fontsize=9)
    plt.axis('off')
    plt.imshow(image.astype(np.uint8), cmap=cmap,
               norm=norm, interpolation=interpolation)
    i += 1
  output_file = os.path.normpath(os.path.join(output_dir, filename))
  plt.savefig(output_file)
  plt.show()

def data_balance_stats(data, title=None, ylabel=None, output_dir=".", filename="databalance_stats.png"):
  """
  Args:
    data : can be a list, np.ndarray or df
  """

  df_sns = pd.DataFrame()
  df_sns['info'] = data

  # if isinstance(data, list):
  #   df_sns = pd.DataFrame()
  #   df_sns['info'] = data
  # elif isinstance(data, np.ndarray):
  #   df_sns = pd.DataFrame()
  #   df_sns['info'] = data
  # elif isinstance(data, pd.DataFrame):
  #   df_sns = pd.DataFrame()
  #   df_sns['info'] = data
  # else:
  #   raise ValueError('class_balance_info waits for list, numpy array or dataframe')

  sns.set_color_codes("pastel")
  sns.set(style="whitegrid")

  # plot
  plt.rcParams["figure.figsize"] = [10.00, 5.50]
  plt.rcParams["figure.autolayout"] = True
  ax0 = plt.subplot(111)

  sns.countplot(y="info", data=df_sns, palette="Set2", ax=ax0)
  if (title is None):
    title = 'Data Info'
  if (ylabel is None):
    ylabel = ''

  ax0.set_title(title)
  ax0.set(xlabel='Number of images', ylabel=ylabel)
  output_file = os.path.normpath(os.path.join(output_dir, filename))
  plt.savefig(output_file)
  plt.show()

def image_size_stats_from_values(df_width, df_height, output_dir=".", filename="image_sizes.png"):
  """
  Args:
    list_files : list of files
  """

  # img_size = [Image.open(im).size for im in list_files]  # im.width, im.height

  widths = []
  heights = []
  shape_freqs = []
  img_shapes_keys = {}
  for height, width in zip(df_width, df_height):
    key = str(height) + '-' + str(width)
    if key in img_shapes_keys:
      shape_id = img_shapes_keys[key]
      shape_freqs[shape_id] += 1
    else:
      img_shapes_keys[key] = len(widths)
      widths.append(width)
      heights.append(height)
      shape_freqs.append(1)

  d = {'Image width (px)': widths, 'Image height (px)': heights, '# images': shape_freqs}
  df = pd.DataFrame(d)
  cmap = sns.cubehelix_palette(dark=.1, light=.6, as_cmap=True)
  plot = sns.scatterplot(x="Image width (px)", y="Image height (px)", size='# images', hue="# images", palette=cmap,
                         data=df)
  plot = plot.set_title('Number of images per image shape', fontsize=15)
  output_file = os.path.normpath(os.path.join(output_dir, filename))
  plt.savefig(output_file)

  plt.show()

def show_images_from_df(df_path_dir, channels=3, output_dir=".", filename="image_sample.png"):
  list_files = df_path_dir.tolist()
  list_files = [Path(i) for i in list_files]
  image = [random.choice(list_files) for im in range(9)]
  label = [f.parents[0].stem for f in image]

  cmap_color = 'viridis'
  if channels == 1:
    cmap_color = 'gray'
  plt.rc('font', size=10)

  i = 0
  for img, lbl in zip(image, label):
    plt.subplot(3, 3, i + 1)
    im = Image.open(str(img))

    # if(max(im)>1.):
    #    plt.imshow(im/255.,cmap=cmap_color)
    # else:
    plt.imshow(im, cmap=cmap_color)
    plt.title(lbl + ' : ' + str(im.size))
    plt.axis(False)
    i += 1
  output_file = os.path.normpath(os.path.join(output_dir, filename))
  plt.savefig(output_file)
  plt.show()


def build_dataset(X, y,
                  is_training,
                  is_augm=False,
                  channels=3,
                  scale=True,  # -_-' no scaling for EfficientNet from keras.applications!
                  img_size=224,
                  data_from="files",  # files, tfrecords, (pixels) values
                  batch_size=32,
                  drop_remainder=False):
  """Retourne un dataset tensorflow (Dataset.from_tensor_slices)
     1. from_tensor_slices
     2. normalisation (load + resize)
     3. shuffle (only if is_training)
     4. batch
     5. data augmentation (only if is_training & is_augm)
     return dataset (no prefetch)
  """

  # 1. from_tensor_slices
  ds = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))

  # 2. normalisation (load + resize)
  dataset = ds.map(lambda image, label: (normalize_img(image, img_size, channels, data_from, scale), label))

  # 3. shuffle
  if is_training:
    num_samples = min(1000, len(y))
    dataset = dataset.shuffle(num_samples)  # num.samples

  # 4. batch
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)


  # 5. data augmentation
  if (is_training and is_augm):
    dataset = dataset.map(lambda image, label: (data_augmentation(image), label))

  dataset = dataset.repeat(count=3)

  return dataset

def resize_img(img, img_size):
  """Fonction de resize qui retient le ratio pour l'appliquer au key points"""
  # old_size = tf.shape(img)[:2]
  img_size = tf.constant(img_size)
  im = tf.image.resize(img, img_size)
  # ratio = tf.math.divide(img_size,old_size)
  return im  # ratio


def normalize_img(input_image, img_size, channels, data_from, scale):
  """Obtain the image from the filename (for both training and validation).

  The following operations are applied:
      1. read & decode : Decode the image from jpeg/png format
      2. resize : change size of image
      3. cast : convert to float
      4. scale : Convert to range [0, 1]
  """
  # 1. read & decode : Decode the image from jpeg/png format
  if (data_from == "files" or data_from == "tfrecords"):
    image_string = tf.io.read_file(input_image)
    image = tf.cond(
      tf.image.is_jpeg(image_string),
      lambda: tf.image.decode_jpeg(image_string, channels=3),
      lambda: tf.image.decode_png(image_string, channels=3))
    # image = tf.image.decode_jpeg(image_string, channels=channels)
    # image = tf.image.decode_png(image_string, channels=channels)
  else:  # attention ici ce sont les values des pixels [0,255]
    image = input_image

  # 2. resize : change size of image
  image = tf.image.resize(image, [img_size, img_size])

  # 3. cast : convert to float
  image = tf.cast(image, tf.float32)

  # 4. scale : Convert to range [0, 1]
  if scale:
    image = image / 255.
  return image


def data_augmentation(X):
  """Data augmentation of training images

  The following operations are applied randomly:
      1. flip_up_down
      2. flip_left_righ
      3. identity
      4. rgb_to_grayscale
  """

  def V_Flip(X):
    X = tf.image.flip_up_down(X)
    return X

  def H_Flip(X):
    X = tf.image.flip_left_right(X)
    return X

  def Contrast(X):
    X = tf.image.random_contrast(X, lower=0.0, upper=1.0)
    return X

  def Brightness(X):
    X = tf.image.random_brightness(X, 0.2)
    return X

  def Identity(X):
    return X

  def GreyScale(X):
    X = tf.image.rgb_to_grayscale(X)
    X = tf.stack([X, X, X], 2)[:, :, :, 0]
    return X

  # Utilisation des outils random de tf pour bien que ca fonctionne avec le format des tenseurs symboliques
  p = tf.random.uniform(shape=[1], minval=0, maxval=5, dtype=tf.dtypes.int32)
  X = tf.cond(p == 0, lambda: Identity(X), lambda: Identity(X))
  X = tf.cond(p == 1, lambda: H_Flip(X), lambda: Identity(X))
  X = tf.cond(p == 2, lambda: V_Flip(X), lambda: Identity(X))
  X = tf.cond(p == 3, lambda: Contrast(X), lambda: Identity(X))
  X = tf.cond(p == 4, lambda: Brightness(X), lambda: Identity(X))
  #X = tf.cond(p == 3, lambda: GreyScale(X), lambda: Identity(X))
  return X

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()

def plot_loss_curves_mae(history):
  """
  Returns separate loss curves for training and validation metrics.
  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  mae = history.history['mae']
  val_mae = history.history['val_mae']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, mae, label='training_mean_absolute_error')
  plt.plot(epochs, val_mae, label='val_mean_absolute_error')
  plt.title('mean_absolute_error')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()