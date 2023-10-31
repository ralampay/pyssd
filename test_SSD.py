import cv2
import torch
import sys
import os
from pyssd.lib import vgg16
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pyssd.lib.ssd300 import SSD300
from pyssd.lib.vgg16 import VGGBase

# ... (previous code)

# Your tensor representing locs and predictions
locs_tensor = torch.tensor([[[-1.2596e+00, -1.3815e+00, 3.5383e-01, 5.1166e-01],
                            [1.8587e-01, 3.0120e-01, -1.2568e-02, 8.0395e-02],
                            [9.2246e-01, -8.0592e-01, -9.5797e-01, 1.0136e+00],
                            [1.9140e-02, 1.9897e-02, 1.6475e-03, -6.8783e-03],
                            [1.8442e-02, -7.1047e-04, -1.8448e-02, 9.0251e-03],
                            [-8.2504e-03, 1.8954e-02, -4.6820e-03, 1.9949e-03]]],
                            dtype=torch.float32)

original_image = cv2.imread("test/sample_training/images/train/00006c07d2b033d1.jpg")
# Resize the image
original_image = cv2.resize(original_image, (300, 300))

# Must be normalized
normalized_image = original_image / 255

transposed_img = normalized_image.transpose((2, 0, 1))

print(f"Original shape: {original_image.shape}")
print(f"Transposed: {transposed_img.shape}")

# Create as tensor
tensor_image = torch.tensor(transposed_img, dtype=torch.float32)
tensor_image = tensor_image.unsqueeze(0)  # Add a batch dimension

print(f"Tensor shape: {tensor_image.shape}")

base = VGGBase()
model = SSD300(1, base)

locs, predictions = model(tensor_image)

# Now replace the locs and predictions with the provided tensor
locs = locs_tensor
predictions = locs_tensor

print(f"Locs Shape: {locs.shape}")
print(f"Predictions Shape: {predictions.shape}")

original_image_width = 300
original_image_height = 300
border_color = (0, 0, 255)
text_color = (0, 0, 0)

for box, class_probs in zip(locs[0], predictions[0]):
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
