import cv2
import torch
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import torch.optim as optim
from pyssd.lib.utils import adjust_learning_rate
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
model = SSD300(n_classes=1, base=base)
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
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
# Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
biases = list()
not_biases = list()
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)
optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    loop = tqdm(train_loader)

    ave_loss = 0.
    count = 0

    # Decay learning rate at particular epochs
    if epoch in decay_lr_at:
        adjust_learning_rate(optimizer, decay_lr_to)

    for batch_idx, (images, boxes, labels) in enumerate(loop):
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        locs, predictions = model(images) 

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

