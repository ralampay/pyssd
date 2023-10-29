import cv2
import torch
from torch.utils.data import Dataset
import os

class CustomImageTextDataset(Dataset):
    def __init__(self, image_path, labels_path, img_dim=(300,300)):
        self.image_path     = image_path
        self.labels_path    = labels_path
        self.img_dim        = img_dim

        self.images = sorted(os.listdir(self.image_path))
        self.labels = sorted(os.listdir(self.labels_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file_loc = os.path.join(self.image_path, self.images[index])
        label_file_loc = os.path.join(self.labels_path, self.labels[index])

        img = cv2.imread(image_file_loc)
        img = cv2.resize(img, self.img_dim)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255

        # transpose
        img = img.transpose((2, 0, 1))

        labels = []
        boxes = []

        with open(label_file_loc, 'r') as label_file:
            for line in label_file:
                values = line.strip().split()

                class_id    = int(values[0])
                x_center    = float(values[1])
                y_center    = float(values[2])
                width       = float(values[3])
                height      = float(values[4])

                labels.append(class_id)

                boxes.append([
                    x_center,
                    y_center,
                    width,
                    height
                ])

        # Convert to tensor
        img = torch.Tensor(img)
        boxes = torch.Tensor(boxes)
        labels = torch.Tensor(labels)

        return img, boxes, labels
