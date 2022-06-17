from src import utils
from helper_functions import CocoLikeDataset
import json
import numpy as np
import argparse
from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
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
from src.utils import data_balance_stats, image_size_stats_from_values, show_images_from_df
from src.utils import display_top_masks,display_images

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./datasets/container',
                    help="Directory containing the dataset")
parser.add_argument('--json_file', default=None,
                    help="name of json file")
parser.add_argument('--output_dir', default='data_inspection',
                    help="Optional, name of output directory")
start_time = time.time()

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


if __name__ == '__main__':
  # Load the parameters from json file
  args = parser.parse_args()

  select = args.json_file
  json_file = 'via_conteneur_coco_' + select + '.json'
  # repertoire des donnees
  path_dir = Path(r'F:\Deep_Learning\SEGA\datasets') #Path(args.data_dir)
  output_dir = "./data_inspection"+"/"+ select
  image_dir = path_dir / 'container'
  json_path = path_dir / 'container' / json_file

  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

  # Set the logger
  logger = ".".join([select,"log"])
  utils.set_logger(os.path.join(args.output_dir, logger))

  # ===========================================================================
  #  Preparation du dataset
  # ===========================================================================
  dataset_train = CocoLikeDataset()
  print(Path(json_path))
  dataset_train.load_data(json_path, image_dir)
  dataset_train.prepare()
  if (len(dataset_train.image_ids) != 0):
    logging.info(f"Nombre d'images dans le dataset: {len(dataset_train.image_ids)}")
  else:
    raise Exception("Le nombre d'images du train set est nul")

  nr_classes = dataset_train.num_classes
  logging.info(f"Nombre de classes dans le dataset: {nr_classes}")

  logging.info("--- ELAPSED TIME : %s seconds ---" % (time.time() - start_time))


  dataset = dataset_train
  # ===========================================================================
  # Visualize Annotated Images (with file path)
  # ===========================================================================

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
      to_display = utils.display_top_masks(image, mask, class_ids, bbox,booldisp,output_dir)
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

  # ===========================================================================
  # dataframe treatment
  # ===========================================================================


  data_balance_stats(df['mode'], title='Mode Balance Info', ylabel='modes', output_dir=output_dir,
                         filename="mode_balance.png")

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