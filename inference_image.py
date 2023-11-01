from torchvision import transforms
from pyssd.lib.utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()
# image_offset = 100
# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./OpenSans-Light.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        box_location = det_boxes[i].tolist()
        x_center, y_center, width, height = box_location
        left = x_center - (width / 2)
        top = y_center - (height / 2)
        right = x_center + (width / 2)
        bottom = y_center + (height / 2)

        # update box_location
        box_location = [left, top, right, bottom]
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        # text_size = font.getsize(det_labels[i].upper())
        # text_size = 1
        text_location = [box_location[0] + 2., box_location[1]]
        textbox_location = [box_location[0], box_location[1] , box_location[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


def draw_many(original_image):
    border_color = (0, 0, 255)
    text_color = (0, 0, 0)
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # Initialize a list to store bounding boxes and labels
    bounding_boxes = []

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height])[0].unsqueeze(0)
    
    for box, class_probs in zip(predicted_locs[0], predicted_scores[0]):
        class_id = class_probs.argmax().item()
        class_probability = class_probs[class_id].item()
        box = box * original_dims
        x_center, y_center, width, height = box.tolist()
        left = x_center - (width / 2)
        top = y_center - (height / 2)
        right = x_center + (width / 2)
        bottom = y_center + (height / 2)


        if left >= 0 and left <= 1024 and top >= 0 and top <= 1024 and right >= 0 and right <= 1024 and bottom >= 0 and bottom <= 1024:
            bounding_boxes.append((left, top, right, bottom, class_id, class_probability))

    # Create a drawing context
    draw = ImageDraw.Draw(original_image) 
    # Define a font for text labels
    font = ImageFont.truetype("./OpenSans-Light.ttf", 12)

    # Now, you have a list of bounding boxes and labels, you can draw them outside the loop
    for box in bounding_boxes:
        x1, y1, x2, y2, class_id, class_probability = box
        draw_box = [x1, y1, x2, y2]
        # Draw the bounding box on the output image
        draw.rectangle(draw_box, outline=border_color)

        # Display class name and probability
        class_name = f'Class {class_id}'
        text = f'{class_name}: {class_probability:.2f}'
        draw.text((x1, y1 - 10), text, fill=text_color, font=font)

    # Display or save the image
    original_image.show()


if __name__ == '__main__':
    img_path = 'test/sample_training/images/train/00006c07d2b033d1.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    # detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
    draw_many(original_image=original_image)
