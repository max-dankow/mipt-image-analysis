import numpy as np
from .image_utils import with_normalised


# Autocontrast


def autocontrast(image, white_perc, black_perc):
    return with_normalised(image, lambda image: __autocontrast(image, white_perc, black_perc))


def __autocontrast(norm_image, white_perc, black_perc):
    # Unfoturnately, method quantile() is not found in numpy
    i_max = np.percentile(norm_image, (1.0 - white_perc) * 100)
    i_min = np.percentile(norm_image, black_perc * 100)
    result = (norm_image - i_min) / (i_max - i_min)
    result[result > 1.0] = 1
    result[result < 0.0] = 0
    return result
