import sys
import os
import platform
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from pyssd.lib.vgg16 import VGGBase
from pyssd.lib.ssd300 import SSD300
from pyssd.lib.custom_image_text_dataset import CustomImageTextDataset
from pyssd.lib.multi_box_loss import MultiBoxLoss
from pyssd.lib.utils import save_checkpoint

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class Train:
    def __init__(self, params):
        self.params = params

        self.device         = params.get('device')
        self.gpu_index      = params.get('gpu_index')
        self.n_classes      = params.get('n_classes')
        self.base           = params.get('base')
        self.epochs         = params.get('epochs')
        self.lr             = params.get('lr')
        self.batch_size     = params.get('batch_size')
        self.labels_path    = params.get('labels_path')
        self.images_path    = params.get('images_path')
        self.model_file     = params.get('model_file')

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)
        elif self.device == 'cpu':
            print(f"CPU Device: {platform.processor()}")

        if self.base == 'vgg16':
            self.base = VGGBase()
        else:
            print(f'Invalid base {self.base}')
            exit(0)

        self.model = SSD300(
            n_classes=self.n_classes, 
            base=self.base
        ).to(self.device)

        self.criterion = MultiBoxLoss(
            priors_cxcy=self.model.priors_cxcy
        ).to(self.device)

        self.dataset = CustomImageTextDataset(
            self.images_path, 
            self.labels_path
        )

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr
        )

    def execute(self):
        print(f"Training for {self.epochs} epochs")
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}")

            loop = tqdm(self.train_loader)

            ave_loss = 0.
            count = 0

            for batch_idx, (images, boxes, labels) in enumerate(loop):
                # Move to default device
                images = images.to(self.device)  # (batch_size (N), 3, 300, 300)

                boxes = boxes.to(self.device)
                labels = labels.to(self.device)
                
                locs, predictions = self.model(images) 

                loss = self.criterion(locs, predictions, boxes, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update tqdm
                loop.set_postfix(loss=loss.item())
                
                ave_loss += loss.item()
                count += 1

            ave_loss = ave_loss / count
            print(f'Ave Loss: {ave_loss}')

            print(f"Saving to {self.model_file}...")
            save_checkpoint(
                epoch, 
                self.model, 
                self.optimizer,
                filename=self.model_file
            )