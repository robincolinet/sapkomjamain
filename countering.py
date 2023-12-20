from rembg import remove
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def resize_and_center_on_white_background(image_backgroundless, new_size=(256, 256)):
    # image_with_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_with_alpha=np.array(image_backgroundless)
    if image_with_alpha.shape[2] != 4:
        raise ValueError("L'image n'a pas de canal alpha")

    ratio = min(new_size[0] / image_with_alpha.shape[1], new_size[1] / image_with_alpha.shape[0])
    new_dimensions = (int(image_with_alpha.shape[1] * ratio), int(image_with_alpha.shape[0] * ratio))

    resized_image_with_alpha = cv2.resize(image_with_alpha, new_dimensions, interpolation=cv2.INTER_AREA)

    white_background = np.ones((new_size[1], new_size[0], 3), dtype=np.uint8) * 255

    x_offset = (new_size[0] - new_dimensions[0]) // 2
    y_offset = (new_size[1] - new_dimensions[1]) // 2

    y1, y2 = y_offset, y_offset + new_dimensions[1]
    x1, x2 = x_offset, x_offset + new_dimensions[0]

    alpha_s = resized_image_with_alpha[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    white_background[y1:y2, x1:x2] = white_background[y1:y2, x1:x2] * alpha_l[:, :, np.newaxis] + \
                                     resized_image_with_alpha[:, :, :3] * alpha_s[:, :, np.newaxis]
    
    return white_background

input_paths = os.listdir('test_image_headmind')
#list of all the input image paths    
    
def countoring_one_pic(image_path):
    full_path = os.path.join('test_image_headmind', image_path)
    input_image = Image.open(full_path)
    output = remove(input_image)
    white_background_image = resize_and_center_on_white_background(output)
    return white_background_image

# for image in input_paths:
#     cv2.imwrite('test_countour/' + image[:-3] + 'png', cv2.cvtColor(countoring_one_pic(image), cv2.COLOR_RGB2BGR))

for image in input_paths:
    countoring_one_pic(image)

## reste à mettre sous forme de transformer 