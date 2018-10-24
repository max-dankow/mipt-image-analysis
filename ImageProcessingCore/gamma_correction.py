import numpy as np
from .image_utils import with_normalised


# Gamma-correction


def gamma_correction(image, a, gamma):
    # todo: check parameters
    return with_normalised(image, lambda image: __gamma_correction(image, a, gamma))


def __gamma_correction(norm_image, a, gamma):
    return a * (norm_image.copy() ** gamma)
