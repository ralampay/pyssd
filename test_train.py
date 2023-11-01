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
model = SSD300(n_classes=2, base=base)
dataset = CustomImageTextDataset(image_path, labels_path)

epochs = 30

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

#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    loop = tqdm(train_loader)

    ave_loss = 0.
    count = 0

    # Decay learning rate at particular epochs
#    if epoch in decay_lr_at:
#        adjust_learning_rate(optimizer, decay_lr_to)

    for batch_idx, (images, boxes, labels) in enumerate(loop):
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
#        boxes = [b.to(device) for b in boxes]
#        labels = [l.to(device) for l in labels]

        boxes = boxes.to(device)
        labels = labels.to(device)
        
        locs, predictions = model(images) 

#        print('locs: ', locs)
#        print('predictions: ', predictions)

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

    print(f'Ave Loss: {ave_loss}')

# Test the model on training data to see if we have something close
original_image = cv2.imread("test/sample_training/images/train/00006c07d2b033d1.jpg")
# Resize the image
original_image = cv2.resize(original_image, (300, 300))
normalized_image = original_image / 255

transposed_img = normalized_image.transpose((2, 0, 1))

print(f"Original shape: {original_image.shape}")
print(f"Transposed: {transposed_img.shape}")

# Create as tensor
tensor_image = torch.tensor(transposed_img, dtype=torch.float32)
tensor_image = tensor_image.unsqueeze(0)    # Add a batch dimension

print(f"Tensor shape: {tensor_image.shape}")

original_image_width = 300
original_image_height = 300

# Set model for evaluation
model.eval()

locs, predictions = model(tensor_image)

print(f"Locs Shape: {locs.shape}")
print(f"Predictions Shape: {predictions.shape}")

original_image_width = 300
original_image_height = 300
border_color = (0, 0, 255)
text_color = (0, 0, 0)


for box, class_probs in zip(locs[0], predictions[0]):
    # Extract center coordinates (x_center, y_center, width, height)
    x_center, y_center, width, height = box

    class_id = class_probs.argmax().item()
    class_probability = class_probs[class_id].item()

    # Calculate the (x1, y1) and (x2, y2) coordinates of the bounding box
    x1 = int((x_center - width / 2) * original_image_width)
    y1 = int((y_center - height / 2) * original_image_height)
    x2 = int((x_center + width / 2) * original_image_width)
    y2 = int((y_center + height / 2) * original_image_height)

    print(f'class_id: {class_id} class_probability: {class_probability}')

    # Only print valid coordinates
    if x1 >= 0 and x1 <= 300 and x2 >= 0 and x2 <= 300 and y1 >= 0 and y1 <= 300 and y2 >= 0 and y2 <= 300:
        # Draw the bounding box on the output image
        print((x1, y1))
        print((x2, y2))

        cv2.rectangle(original_image, (x1, y1), (x2, y2), border_color, 1)

        # Display class name and probability
        class_name = f'Class {class_id}'
        text = f'{class_name}: {class_probability:.2f}'
        cv2.putText(original_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)


# Display or save the image
cv2.imshow('SSD Output', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
