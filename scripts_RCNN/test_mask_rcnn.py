import os
import sys
import random
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Root directory of the project
ROOT_DIR = './weights'
RESULTS_DIR = './rcnn_outputs'
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import src.mrcnn_38.utils as utils
from src.mrcnn_38 import visualize
import src.mrcnn_38.model as modellib
from src.mrcnn_38.container import ContainerConfig, CocoLikeDataset


def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in
  all visualizations in the notebook. Provide a
  central point to control graph sizes.

  Adjust the size attribute to control how big to render images
  """
  _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
  return ax


def evaluate_coco(model, dataset, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "covid_mask_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Limit to a subset
    if limit:
        #image_ids= dataset.image_ids[:limit]
        image_ids = [random.choice(dataset.image_ids) for i in range(limit)]
    else:
        image_ids = dataset.image_ids  # image_ids or


    # Get corresponding TACO image IDs.
    taco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()
    results = []
    APs = dict()
    for i, image_id in enumerate(image_ids):
        # Info image
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        # # Load image
        # image = dataset.load_image(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)


        # Run detection
        t = time.time()
        # r = model.detect([image], verbose=0)[0]
        results = model.detect([image], verbose=1)
        t_prediction += (time.time() - t)

        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions",show_mask=False)
        scores = r["scores"]
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

        # Draw precision-recall curve
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.8)
        print("id - AP",dataset.image_info[image_id]["id"],AP)
        APs[dataset.image_info[image_id]["id"]] = {'AP'       : AP,
                                                   'gt_bbox'  : gt_bbox,
                                                   'pred_bbox': r['rois'],
                                                   'score'    : r['scores']}
        print(APs)
        visualize.plot_precision_recall(AP, precisions, recalls)

        # Grid of ground truth objects and their predictions
        visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                                overlaps, dataset.class_names)

        #plt.show()

    # Save to csv file
    # APs = "ImageId,EncodedPixels\n" + "\n".join(APs)
    file_path = os.path.join(submit_dir, "APs_covid_mask.csv")
    df_APs = pd.DataFrame(columns=['image_id','AP','gt_bbox','pred_bbox','score'])
    for img_id, value in APs.items():
        line = [img_id, value['AP'],value['gt_bbox'],value['pred_bbox'],value['score']]
        print(line)
        df_APs.loc[len(df_APs.index)] = line

    # with open(file_path, "w") as f:
    #     f.write(str(APs))
    # boucle sur chaque bbox
    # for i in r['rois'].shape[0]:
    #     df_APs['image_id'] = gt_bbox[i]
    #     df_APs['gt_bbox']  = gt_bbox[i]
    #     df_APs['pred_bbox']= r['rois'][i]
    #     df_APs['pred_bbox']= r['scores'][i]
    df_APs.to_csv(file_path)
    print("Saved to ", submit_dir)

    print(df_APs)
    print("mAP @ IoU=50: ", df_APs['AP'].mean())     #np.mean(APs))
    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run test Mask R-CNN.')
    parser.add_argument('--model_log', required=False, default='last', type=str, help='path to model log')
    args = parser.parse_args()

    config = ContainerConfig()
    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        USE_MINI_MASK = False

    config = InferenceConfig()
    config.display()

    # Load Validation Dataset
    dataset_test = CocoLikeDataset()
    # json_file = './datasets/split_dataset3/annotations/coco_annotations_test.json'
    # img_dir = Path('./datasets/split_dataset3/images/test')
    json_file = './datasets/container/coco_annotations_test.json'
    img_dir = Path('./datasets/container')
    if (not img_dir.exists()):
        raise AssertionError("Images directory is not found at {}".format(img_dir))
    dataset_test.load_data(json_file, img_dir)
    dataset_test.prepare()
    if (len(dataset_test.image_ids)!=0):
        print("Nombre d'images dans le test set :", len(dataset_test.image_ids))
    else:
        raise Exception("Le nombre d'images du test set est nul")
    nr_classes = dataset_test.num_classes
    print("Nombre de classes dans le train set :", nr_classes)


    # Load Model
    # Directory to save logs and trained model
    model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR,config=config)
    if (args.model_log=="last"):
        model_path = model.find_last()
        print(model_path)
    else:
        model_path = args.model_log

    print(model_path)
    assert os.path.exists(model_path), 'model_path does not exist. Did you forget to read the instructions above? ;)'
    # # si on veut loader un autre fichier
    # model_path = './weights/logs/container_model20220626T1808'

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Run Detection
    nr_eval_images = len(dataset_test.image_ids)
    print("Running COCO evaluation on {} images.".format(nr_eval_images))
    evaluate_coco(model, dataset_test, "bbox", limit=10)
