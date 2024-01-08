from rembg import remove
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def resize_and_center_on_white_background(image_backgroundless, new_size=(256, 256)):
    """
    Resize the image while maintaining its aspect ratio and center it on a white background.
    Parameters:
    - image_backgroundless (PIL.Image): Image with a transparent background.
    - new_size (tuple, optional): New size for the image on the white background.
    Returns:
    numpy.ndarray: Image with a white background, resized and centered.
    """
    image_with_alpha = np.array(image_backgroundless)
    if image_with_alpha.shape[2] != 4:
        raise ValueError("The image does not have an alpha channel")

    # Calculate the resizing ratio
    ratio = min(new_size[0] / image_with_alpha.shape[1], new_size[1] / image_with_alpha.shape[0])
    new_dimensions = (int(image_with_alpha.shape[1] * ratio), int(image_with_alpha.shape[0] * ratio))

    # Resize the image with alpha channel
    resized_image_with_alpha = cv2.resize(image_with_alpha, new_dimensions, interpolation=cv2.INTER_AREA)

    # Create a white background
    white_background = np.ones((new_size[1], new_size[0], 3), dtype=np.uint8) * 255

    # Calculate offset to center the resized image on the white background
    x_offset = (new_size[0] - new_dimensions[0]) // 2
    y_offset = (new_size[1] - new_dimensions[1]) // 2

    y1, y2 = y_offset, y_offset + new_dimensions[1]
    x1, x2 = x_offset, x_offset + new_dimensions[0]

    # Blend the resized image with the white background using the alpha channel
    alpha_s = resized_image_with_alpha[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    white_background[y1:y2, x1:x2] = white_background[y1:y2, x1:x2] * alpha_l[:, :, np.newaxis] + resized_image_with_alpha[:, :, :3] * alpha_s[:, :, np.newaxis]
    
    return white_background


input_paths = os.listdir('data/test_image_headmind')
    

def countoring_one_pic(image_path):
	"""
    Open an image, remove the background using the rembg library,
    and prepare the resulting image for display.
    Parameters:
    - image_path (str): Path to the input image.
    Returns:
    numpy.ndarray: Processed image with a white background.
    """
    full_path = os.path.join('data/test_image_headmind', image_path)
    input_image = Image.open(full_path)
    output = remove(input_image)
    white_background_image = resize_and_center_on_white_background(output)
    return white_background_image


if __name__ == '__main__':
	for image in input_paths[:5]:    
		plt.imshow(countoring_one_pic(image))
		plt.axis('off')
		plt.show()
