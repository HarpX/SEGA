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