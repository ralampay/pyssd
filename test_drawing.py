import cv2
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

original_image = cv2.imread("test/sample_training/images/train/00006c07d2b033d1.jpg")
# Resize the image
original_image = cv2.resize(original_image, (300, 300))

original_image_width = 300
original_image_height = 300
border_color = (0, 0, 255)
text_color = (0, 0, 0)

# Read the data
file_path = "test/sample_training/test_output.txt"
with open(file_path, "r") as file:
    for line in file:
        values = line.strip().split()

        class_id    = values[0]
        x_center    = float(values[1])
        y_center    = float(values[2])
        width       = float(values[3])
        height      = float(values[4])

        # Calculate the (x1, y1) and (x2, y2) coordinates of the bounding box
        x1 = int((x_center - width / 2) * original_image_width)
        y1 = int((y_center - height / 2) * original_image_height)
        x2 = int((x_center + width / 2) * original_image_width)
        y2 = int((y_center + height / 2) * original_image_height)

        # Draw the bounding box on the output image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), border_color, 1)

 
        # Display class name and probability
        class_probability = 1
        class_name = f'Class {class_id}'
        text = f'{class_name}: {class_probability:.2f}'
        cv2.putText(original_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

# Display or save the image
cv2.imshow('SSD Output', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
