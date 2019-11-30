import tensorflow as tf
from matplotlib.image import imread
import matplotlib.pyplot as plt
from PIL import Image

def make_square(im, fill_color=(0, 0, 0)):
    """
    Makes the given PIL Image object a square one, with the
    extra pixels filled in with the provided fill_color
    """
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def resize(image, size):
    """
    Resizes the given PIL Image object to a given (square) size. Fills
    in the spaces of rectangular images with black
    """
    image_matrix = make_square(image)
    image_matrix = image_matrix.resize((size, size), Image.NONE)
    return image_matrix
