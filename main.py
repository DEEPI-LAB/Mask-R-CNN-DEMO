#!/usr/bin/env python
# coding: utf-8
import colorsys
import glob
import os
import sys
import random
import numpy as np
import cv2
from skimage.measure import find_contours
from samples.coco import coco
from mrcnn import utils
import mrcnn.model as modellib
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

class InferenceConfig(coco.CocoConfig):
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)

#
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# # Run Object Detection
# webcam
camera = cv2.VideoCapture(0)
# image
# image_dir = glob.glob('./images/*.jpg')
# item = 0
while True:
    ret, image = camera.read()
    # image = cv2.imread(camera)
    # image = cv2.imread(image_dir[item])
    # item = item + 1
    if ret is True:

        # Load images from the images folder or Webcam
        try:
            image = cv2.resize(image,dsize=None,fx=0.5,fy=0.5)
        except:
           print("error")
        # Run detection
        results = model.detect([image], verbose=2)
        # Visualize results
        r = results[0]

        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        scores = r['scores']
        N = boxes.shape[0]
        colors = random_colors(N, True)

        for i in range(N):
            color = colors[i]
            y1, x1, y2, x2 = boxes[i]
            # insert bounding boxes
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255),
                          cv2.INTER_LINEAR)
            # Mask Polygon
            mask = masks[:, :, i]
            image = apply_mask(image, mask, color)
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            # insert class name
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
            cv2.putText(image, caption, (x1, y1 - 15), cv2.FONT_ITALIC, 0.8, (255, 255, 255))

            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = np.around(verts)
                cv2.polylines(image, np.int32([verts]), False, (0, 255, 255))
        try:
            cv2.imshow("Mask R-CNN",image)
            if cv2.waitKey(100) > 0:
                break
        except:
            print("error")