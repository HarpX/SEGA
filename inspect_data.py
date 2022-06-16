import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa
from pathlib import Path
import pandas as pd
import argparse
from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab
import logging
import os
import random
import cv2
import math
from PIL import Image, ImageStat
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
#import tensorflow as tf
from my_helper_function import walk_through_dir,show_images_from_df
from my_helper_function import data_balance_stats,image_size_stats,show_images_from_dir,build_dataset
from my_helper_function import show_images_from_ds,split_dataset2,model_func_conv,model_compile_classification
from my_helper_function import model_fit,model_seq_efficientnet,compare_histories
from my_helper_function import model_seq_resnet,model_func_resnet_upgrade,model_seq_densenet
from my_helper_function import predict_from_ds,calculate_scores
from my_helper_function import make_confusion_matrix,results_to_frame,restore_model,save_model
from my_helper_function import plot_components,from_filenames_to_nparray,show_images_from_values
from my_helper_function import compute_and_plot_isomap,rgb2gray, compute_and_plot_pca
from my_helper_function import data_to_list_labels,data_to_list_files,seuillage_otsu_and_plot
from mrcnn_38.cig_and_mask import CocoLikeDataset
from mrcnn_38 import visualize
from mrcnn_38.utils import set_logger
from my_helper_function import image_size_stats_from_values
from mrcnn_38.visualize import display_images,draw_boxes

select = 'train'
json_ext = 'via_conteneur_coco_'+select+'.json'
# repertoire des donnees
path_dir = Path(r'F:\Projet_Datascientest\dataset')
output_dir ="./"+select+"_set_statistics"
image_dir = path_dir/'container'
json_train_file = path_dir/'container'/json_ext

if not os.path.exists(output_dir):
  os.mkdir(output_dir)
  overwrite = 'y'
else:
  print("Warning: dir {} already exists. Overwrite".format(output_dir))

# Set the logger
logger = ".".join([select,"log"])
set_logger(os.path.join(output_dir,logger))

def data_to_frame(path_dir):
  """
  Convert data in rep_dir to frame
  Args:
    dir_path (str): target directory

  Returns:
    - list of names of classes (categories) read from path
    - a DataFrame :
      path   |   extension   |   size   |   label_name   |   label_int
  """
  # recuperation des listes de fichiers, labels, extensions de fichier
  files, extensions = data_to_list_files(path_dir, extension=True)
  # recuperation des tailles des images
  img_size = [Image.open(im).size for im in files]  # im.width, im.height

  # on passe tout dans un DataFrame
  df = pd.DataFrame({"path": files,
                     "extension": extensions,
                     "size": img_size
                     })

  # print des infos
  #logging.info(f"Total number of images : {len(df['path'])} ")
  return  df


# arborescence du rep des donnees
for dirpath, dirnames, filenames in os.walk(path_dir):
  logging.info(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")

files = data_to_list_files(path_dir, extension=False)
logging.info(f"The dataset contains {len(files)} images in '{dirpath}'.")

# classes presentes
train_dir = path_dir
df2 = data_to_frame(train_dir)
print(df2)
#
class_names, label_names, label_int = data_to_list_labels(train_dir)
# data_balance_stats(label_names,title='Data Balance Info',ylabel='Categories', output_dir=output_dir,filename="data_balance.png")
# data_balance_stats(extensions,title='Extension Balance Info',ylabel='Extensions', output_dir=output_dir,filename="extension_balance.png")
# image_size_stats(file_list,output_dir=output_dir,filename="image_size.png")
# show_images_from_dir(train_dir,output_dir=output_dir,filename="image_sample.png")


# on commence par le json du train
with open(json_train_file, 'r') as f:
  dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)
logging.info(f'Number of categories: {nr_cats}')
logging.info(f'Number of annotations: {nr_annotations}')
logging.info(f'Number of images:{nr_images} --> {nr_images/len(files)*100} % of the whole dataset')



# ===========================================================================
# 3. Preparation du dataset
# ===========================================================================
# ---------------------------
# on cree le dataset image+annotation
dataset_train = CocoLikeDataset()
dataset_train.load_data(json_train_file,image_dir)
dataset_train.prepare()
if (len(dataset_train.image_ids) != 0):
  logging.info(f"Nombre d'images dans le dataset: {len(dataset_train.image_ids)}")
else:
  raise Exception("Le nombre d'images du train set est nul")

nr_classes = dataset_train.num_classes
logging.info(f"Nombre de classes dans le dataset: {nr_classes}")

def rgb2gray(rgb):
  return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# la moyenne de la brightness d'une image (on moy le gray)
def image_brightness(img):
  nr_of_pixels = img.shape[0]*img.shape[1]
  return np.sum(img) / nr_of_pixels

def image_luminance(img):
  nr_of_pixels = img.shape[0]*img.shape[1]
  return np.sum(img) / (3*nr_of_pixels)

def variance_of_laplacian(img):
  return cv2.Laplacian(img, cv2.CV_64F).var()

def brightness(im_file):
  im = Image.open(im_file)
  stat = ImageStat.Stat(im)
  r, g, b = stat.mean
  return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

dataset = dataset_train
# =======================================================
# ===========================================================================
# 3. Visualize Annotated Images (with file path)
# ===========================================================================

# User settings
image_ids = np.random.choice(dataset.image_ids, 2)
i =0
for image_id in image_ids:
  image_filename=dataset.image_info[image_id]['name']
  image_filepath=dataset.image_info[image_id]['path']
  pylab.rcParams['figure.figsize'] = (28, 28)
  ####################

  # Obtain Exif orientation tag code
  for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
      break

  # Loads dataset as a coco object
  coco = COCO(json_train_file)

  # Find image id
  img_id = -1
  for img in imgs:
    if img['file_name'] == image_filename:
      img_id = img['id']
      break

  # Show image and corresponding annotations
  if img_id == -1:
    print('Incorrect file name')
  else:

    # Load image
    I = Image.open(image_filepath)

    # Load and process image metadata
    if I._getexif():
      exif = dict(I._getexif().items())
      # Rotate portrait and upside down images if necessary
      if orientation in exif:
        if exif[orientation] == 3:
          I = I.rotate(180, expand=True)
        if exif[orientation] == 6:
          I = I.rotate(270, expand=True)
        if exif[orientation] == 8:
          I = I.rotate(90, expand=True)
    # Show image
    fig, ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(I)

    # Load mask ids
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    anns_sel = coco.loadAnns(annIds)

    # Show annotations
    for ann in anns_sel:
      color = colorsys.hsv_to_rgb(np.random.random(), 1, 1)
      for seg in ann['segmentation']:
        poly = Polygon(np.array(seg).reshape((int(len(seg) / 2), 2)))
        p = PatchCollection([poly], facecolor=color, edgecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
      [x, y, w, h] = ann['bbox']
      rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color,
                       facecolor='none', alpha=0.7, linestyle='--')
      ax.add_patch(rect)

    i+=1
    output_file = os.path.normpath(os.path.join(output_dir, 'visualize_annotated_image_'+str(i)+'.png'))
    plt.savefig(output_file)
    plt.show()
    if (i==2):
      break

# ===========================================================================
# 3. Loop sur toutes les images
# ===========================================================================
def over_exposure(im_gray):
  bright_thres = 0.5
  dark_thres = 0.4
  dark_part = cv2.inRange(im_gray, 0, 10)
  bright_part = cv2.inRange(im_gray, 240, 255)
  # use histogram
  # dark_pixel = np.sum(hist[:30])
  # bright_pixel = np.sum(hist[220:256])
  total_pixel = np.size(im_gray)
  dark_pixel = np.sum(dark_part > 0)
  bright_pixel = np.sum(bright_part > 0)
  state = "ok"
  if dark_pixel/total_pixel > dark_thres :
      state = "under-exposed"
  if bright_pixel/total_pixel > bright_thres:
      state = "over-exposed"
  return state


# image_ids = np.random.choice(dataset.image_ids, 2)
im_tot_masque = np.zeros((216,216))
im_tot_gray = np.zeros((216,216))
# for image_id in image_ids:
df = pd.DataFrame(columns=["image_path","image_file","image_id","format","mode","width",
                           "height","nb_masks","masks_area",
                           "area_ratio", "brightness", "luminance",
                           "exposure","laplacian"])

df_error = pd.DataFrame(columns=["image_file","type_error"])
for i,image_id in tqdm(enumerate(dataset.image_ids)):
    #print("on charge le masque de l'image:",image_id)
    try:
        image = dataset.load_image(image_id)
    except ValueError as e :
        df_error.loc[i] = [dataset.image_info[image_id]['name'],"annotation"]
        continue
    format = dataset.image_info[image_id]['format']
    mode = dataset.image_info[image_id]['mode']
    width = dataset.image_info[image_id]['width']
    height = dataset.image_info[image_id]['height']
    # on recupere le mask
    mask, class_ids, bbox = dataset.load_mask(image_id)

    randomlist = random.sample(range(0, len(dataset.image_ids)), 5)
    if (i in randomlist):
      booldisp=i
    else:
      booldisp=0
    to_display = visualize.display_top_masks(image, mask, class_ids, bbox,booldisp,output_dir)
    black = np.where(to_display[2]>0., 1, 0)
    area_tot = np.sum(black)
    le_masque = np.array(to_display[2])/255
    im_masque = cv2.resize(le_masque, (216, 216))
    im_tot_masque = im_tot_masque + im_masque
    # on convertit l'image en gris
    im_rgb = np.array(to_display[0]) / 255
    im_rgb = cv2.resize(im_rgb, (216, 216))
    im_gray = rgb2gray(im_rgb)
    im_tot_gray = im_tot_gray + im_gray
    #bright = image_brightness(im_gray)
    if (mode=="RGB"):
        bright = brightness(dataset.image_info[image_id]['path'])
    else:
      df_error.loc[i] = [dataset.image_info[image_id]['name'], "gray"]
      bright = 0
    # luminance
    luminance = image_luminance(image)
    # over exposure
    exposure = over_exposure(im_gray)
    # detection d'image floue (blur detection)
    laplacian = variance_of_laplacian(im_gray)

    # on remplit le df
    df.loc[i] =  [dataset.image_info[image_id]['path'],
                  dataset.image_info[image_id]['name'],
                  image_id,
                  format,
                  mode,
                  width,
                  height,
                  mask.shape[-1],
                  area_tot,
                  area_tot/(width*height),
                  bright,
                  luminance,
                  exposure,
                  laplacian
                ]
    #taille de la box dans les deux dimensions pour chaque boite : ratio?

data_balance_stats(df['mode'], title='Mode Balance Info', ylabel='modes', output_dir=output_dir,
                       filename="mode_balance.png")

print(df)

# save csv
output_file = os.path.normpath(os.path.join(output_dir, select+"_info_data.csv"))
df.to_csv(output_file)

data_balance_stats(df['format'],title='Extension Balance Info',ylabel='Extensions', output_dir=output_dir,filename="extension_balance.png")
image_size_stats_from_values(df['width'],df['height'],output_dir=output_dir,filename="image_size.png")
show_images_from_df(df['image_path'],output_dir=output_dir,filename="image_sample.png")


# emplacement des masques (distribution)
plt.imshow(im_tot_masque,cmap="gray")
plt.grid(False)
output_file = os.path.normpath(os.path.join(output_dir, 'mask_distribution.png'))
plt.savefig(output_file)
plt.show()

plt.grid(False)
plt.imshow(im_tot_gray,cmap="gray")
plt.show()

# histogramme area ratio
plt.hist(df["area_ratio"].ravel(), bins=256)
plt.title('Histogram : Mask area (from segmentation) over image size')
plt.grid(False)
output_file = os.path.normpath(os.path.join(output_dir, 'area_ratio.png'))
plt.savefig(output_file)
plt.show()

# hsitogramme luminance
plt.hist(df["luminance"].ravel(), bins=256)
plt.title('Luminance')
plt.grid(False)
output_file = os.path.normpath(os.path.join(output_dir, 'luminance.png'))
plt.savefig(output_file)
plt.show()

# histogramme brightness
plt.hist(df["brightness"].ravel(), bins=256)
plt.title('Brightness')
plt.grid(False)
output_file = os.path.normpath(os.path.join(output_dir, 'brightness.png'))
plt.savefig(output_file)
plt.show()

# histogramme nb d'annotations par image
plt.hist(df["nb_masks"].ravel(), bins=100)
plt.title('Histogram : Number of annotations per image')
plt.grid(False)
output_file = os.path.normpath(os.path.join(output_dir, 'number_annotations.png'))
plt.savefig(output_file)
plt.show()

# traitement des blurred images
plt.hist(df["laplacian"].ravel(), bins=100)
plt.title('Histogram : Variance of Laplacian to detect blurred images')
plt.grid(False)
thresh = 0.01
plt.axvline(thresh, color='r')
output_file = os.path.normpath(os.path.join(output_dir, 'blurred_images.png'))
plt.savefig(output_file)
plt.show()


print(df_error)
output_file = os.path.normpath(os.path.join(output_dir, select+"_suspicious_data.csv"))
df_error.to_csv(output_file)

logging.info(f"Il y a {len(df_error[df_error['type_error']=='gray'])} de photos en echelle de gris")
logging.info(f"Il y a {len(df_error[df_error['type_error']=='annotation'])} de photos avec pb d'annotation")
logging.info(f"{df_error[df_error['type_error']=='annotation']['image_file']} ")
logging.info(f"Il y a {len(df[df['exposure']=='overexposed'])} de photos sur-exposees")
logging.info(f"{df[df['exposure']=='overexposed']['image_file']}")
logging.info(f"Il y a {len(df[df['exposure']=='underexposed'])} de photos sous-exposees")
logging.info(f"{df[df['exposure']=='underexposed']['image_file']}")
logging.info(f"Il y a {len(df[df['brightness']>=180])} de photos avec trop de brightness")
logging.info(f"{df[df['brightness']>=180]['image_file']}")
logging.info(f"Il y a {len(df[df['brightness']<=60])} de photos avec pas assez de brightness")
logging.info(f"{df[df['brightness']<=60]['image_file']}")
logging.info(f"Il y a {len(df[df['laplacian']<=0.01])} de photos suspectee blurred - threshold=0.01")
logging.info(f"{df[df['laplacian']<=0.01]['image_file']}")