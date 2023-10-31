import cv2
import torch
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import torch.optim as optim
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
model = SSD300(1, base).to(device)

dataset = CustomImageTextDataset(image_path, labels_path)

epochs = 5

batch_size = 5

criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

learning_rate = 0.0001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    loop = tqdm(train_loader)

    ave_loss = 0.
    count = 0

    for batch_idx, (img, boxes, labels) in enumerate(loop):
        check1 = img
        check2 = boxes
        check3 = labels
        check4 = torch.squeeze(img)
        shape = img[0].size(0)
        locs, predictions = model(img) 

        print('locs: ', locs)
        print('predictions: ', predictions)

        loss = criterion(locs, predictions, boxes, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm
        loop.set_postfix(loss=loss.item())
        

        print('loss: ', loss.item())
        ave_loss += loss.item()
        count += 1

    ave_loss = ave_loss / count
    print(ave_loss)

