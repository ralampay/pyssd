import sys
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyssd.lib.utils import *

class DetectImage:
    def __init__(self, params):
        self.params = params

        self.device         = params.get('device')
        self.gpu_index      = params.get('gpu_index')
        self.model_file     = params.get('model_file')
        self.image_file     = params.get('image_file')
        self.min_score      = params.get('min_score')
        self.max_overlap    = params.get('max_overlap')
        self.top_k          = params.get('top_k')
        self.suppress       = params.get('suppress')

        self.original_image = Image.open(
            self.image_file,
            mode='r'
        )

        self.original_image = self.original_image.convert('RGB')

        self.resize = transforms.Resize((300, 300))

        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)
        elif self.device == 'cpu':
            print(f"CPU Device: {platform.processor()}")

        # Initialize the model
        self.model = torch.load(self.model_file, map_location=torch.device(self.device))['model']
        self.model.eval()

    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    def execute(self):
        image = self.resize(self.original_image)
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = image.to(self.device)

        # Forward prop.
        predicted_locs, predicted_scores = self.model(image.unsqueeze(0))

        det_boxes, det_labels, det_scores = self.model.detect_objects(
            predicted_locs, 
            predicted_scores, 
            min_score=self.min_score,
            max_overlap=self.max_overlap, 
            top_k=self.top_k
        )

        # Move detections to device
        det_boxes = det_boxes[0].to(self.device)

        # Transform to original image dimensions
        original_dims = torch.FloatTensor([
            self.original_image.width, 
            self.original_image.height, 
            self.original_image.width, 
            self.original_image.height
        ]).unsqueeze(0)

        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to(self.device).tolist()]

        border_color = (0, 0, 255)

        open_cv_image = np.array(self.original_image)
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if self.suppress is not None:
                if det_labels[i] in self.suppress:
                    continue

            box_location = det_boxes[i].tolist()
            x_center, y_center, width, height = box_location

            # Calculate the (x1, y1) and (x2, y2) coordinates of the bounding box
            x1 = int((x_center - width / 2))
            y1 = int((y_center - height / 2))
            x2 = int((x_center + width / 2))
            y2 = int((y_center + height / 2))

            # Draw the bounding box on the output image
            cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), border_color, 1)

        # Display or save the image
        cv2.imshow('SSD Output', open_cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
