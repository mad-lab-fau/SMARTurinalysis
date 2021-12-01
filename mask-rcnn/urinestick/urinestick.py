"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/stick/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/stick/dataset --weights=last
    # Train a new model starting from ImageNet weightss
    python3 coco.py train --dataset=/path/to/stick/dataset --weights=imagenet
    # Apply color splash to an image
    python3 coco.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 coco.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

# Root directory of the project
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
if ROOT_DIR.endswith("urinestick"):
    # Go up one level
    ROOT_DIR = os.path.dirname(ROOT_DIR)
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR + "/mrcnn")
import model as modellib
import utils
from config import Config
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class StickConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "stick"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + stick

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 120

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

    # Number of validation steps
    VALIDATION_STEPS = 47
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
############################################################
#  Dataset
############################################################

class StickDataset(utils.Dataset):

    def load_stick(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("stick", 1, "stick")

        # Train or validation dataset?
        assert subset in ["train", "val", "test", "all"]
        
        # Load annotations
        # VGG Image Annotator: Format of annotations
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
      
        
        if subset == "val":
            annotations = json.load(open(os.path.join(dataset_dir, "00_val_k3.json")))
            print("Working with validation set, Fold 3")
        elif subset == "train":
            annotations = json.load(open(os.path.join(dataset_dir, "00_train_k3.json")))
            print("Working with training set, Fold 3")
        elif subset == "test":
            annotations = json.load(open(os.path.join(dataset_dir, "00_test_k3.json")))
            print("Working with test set, Fold 3")
        elif subset == "all":
            annotations = json.load(open(os.path.join(dataset_dir, "00_all_annotations.json")))

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            if os.path.exists(image_path):
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "stick",
                    # use file name as a unique image id
                    image_id=a['filename'],
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a stick dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "stick":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "stick":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset. with 5 Fold Cross Validation
    epoch_count = 0
    dataset_train = StickDataset()
    dataset_train.load_stick(args.dataset, "train")
        
    dataset_train.prepare()

    # Validation dataset
    dataset_val = StickDataset()
    dataset_val.load_stick(args.dataset, "val")
    dataset_val.prepare()

    #model_inference = modellib.MaskRCNN(
    #    mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
        datetime.datetime.now())
    skimage.io.imsave(file_name, splash)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect the urine sticks.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/stick/dataset/",
                        help='Directory of the Stick dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image,\
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = StickConfig()
    else:
        class InferenceConfig(StickConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
