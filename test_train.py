import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

import sys
import os
from tqdm import tqdm
import torch.optim as optim
from pyssd.lib.utils import adjust_learning_rate, save_checkpoint
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pyssd.lib.vgg16 import VGGBase
from pyssd.lib.auxiliary_convolutions import AuxiliaryConvolutions
from pyssd.lib.prediction_convolutions import PredictionConvolutions
from pyssd.lib.ssd300 import SSD300
from pyssd.lib.custom_image_text_dataset import CustomImageTextDataset
from pyssd.lib.multi_box_loss import MultiBoxLoss

image_path = "test/sample_training/images/train"
labels_path = "test/sample_training/labels/train"

device = "cpu"

base = VGGBase()
model = SSD300(n_classes=2, base=base)
dataset = CustomImageTextDataset(image_path, labels_path)

epochs = 100

batch_size = 5

criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

lr = 1e-3  # learning rate

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    loop = tqdm(train_loader)

    ave_loss = 0.
    count = 0

    for batch_idx, (images, boxes, labels) in enumerate(loop):
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)

        boxes = boxes.to(device)
        labels = labels.to(device)
        
        locs, predictions = model(images) 

        loss = criterion(locs, predictions, boxes, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm
        loop.set_postfix(loss=loss.item())
        

#        print('loss: ', loss.item())
        ave_loss += loss.item()
        count += 1

    ave_loss = ave_loss / count
    save_checkpoint(epoch, model, optimizer)
    print(f'Ave Loss: {ave_loss}')
