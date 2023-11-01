import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class CustomImageTextDataset(Dataset):
    def __init__(self, image_path, labels_path, img_dim=(300,300)):
        self.image_path     = image_path
        self.labels_path    = labels_path
        self.img_dim        = img_dim

        self.images = sorted(os.listdir(self.image_path))
        self.labels = sorted(os.listdir(self.labels_path))

        self.resize = transforms.Resize(img_dim)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file_loc = os.path.join(self.image_path, self.images[index])
        label_file_loc = os.path.join(self.labels_path, self.labels[index])

        original_image = Image.open(image_file_loc, mode='r')
        original_image = original_image.convert('RGB')

        image = self.resize(original_image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        labels = []
        boxes = []

        with open(label_file_loc, 'r') as label_file:
            for line in label_file:
                values = line.strip().split()

                class_id    = int(values[0]) + 1 # Add plus 1 to ensure no 0 value
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
        # img = torch.Tensor(img)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        return image, boxes, labels
