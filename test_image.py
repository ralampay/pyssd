import cv2
import torch
import sys
import os
from pyssd.lib import vgg16
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pyssd.lib.ssd300 import SSD300
from pyssd.lib.vgg16 import VGGBase

original_image = cv2.imread("test/test.png")
# Resize the image
original_image = cv2.resize(original_image, (300, 300))

# Must be normalized
normalized_image = original_image / 255

transposed_img = normalized_image.transpose((2, 0, 1))

print(f"Original shape: {original_image.shape}")
print(f"Transposed: {transposed_img.shape}")

# Create as tensor
tensor_image = torch.tensor(transposed_img, dtype=torch.float32)
tensor_image = tensor_image.unsqueeze(0)    # Add a batch dimension

print(f"Tensor shape: {tensor_image.shape}")

base = VGGBase()
model = SSD300(1, base)

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

    # Draw the bounding box on the output image
    cv2.rectangle(original_image, (x1, y1), (x2, y2), border_color, 1)

    # Display class name and probability
    class_name = f'Class {class_id}'
    text = f'{class_name}: {class_probability:.2f}'
    cv2.putText(original_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)


# Display or save the image
cv2.imshow('SSD Output', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
