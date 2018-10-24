import numpy as np


def convert_to_greyscale(image):
    result = np.copy(image)
    #for pixel in
    return np.mean(image, axis=2).astype(np.uint8)
