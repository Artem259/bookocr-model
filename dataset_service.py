import cv2
import numpy as np
from PIL import Image


def crop_image(image, axis=None):
    points = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(points)
    if axis == 0:
        return image[y:y+h, :]
    if axis == 1:
        return image[:, x:x+w]
    return image[y:y+h, x:x+w]


def make_square(image):
    original_height, original_width = image.shape
    max_dim = max(original_height, original_width)
    expanded_image = np.zeros((max_dim, max_dim), dtype=image.dtype)

    padding_top = (max_dim - original_height) // 2
    padding_left = (max_dim - original_width) // 2
    expanded_image[padding_top:padding_top + original_height, padding_left:padding_left + original_width] = image

    return expanded_image


def resize(image, size):
    image = Image.fromarray(image, mode='L')
    return np.array(image.resize((size, size), resample=Image.BICUBIC))
