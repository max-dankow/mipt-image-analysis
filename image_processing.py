import numpy as np
import image_utils

import image_utils

# Gamma-correction

def gamma_correction(image, a, gamma):
    # todo: check parameters
    return image_utils.with_normalised(image, lambda image: __gamma_correction(image, a, gamma))

def __gamma_correction(norm_image, a, gamma):
    return a * (norm_image.copy() ** gamma)


# Autocontrast

def autocontrast(image, white_perc, black_perc):
    return image_utils.with_normalised(image, lambda image: __autocontrast(image, white_perc, black_perc))

def __autocontrast(norm_image, white_perc, black_perc):
    # Unfoturnately, method quantile() is not found in numpy
    i_max = np.percentile(norm_image, (1.0 - white_perc) * 100)
    i_min = np.percentile(norm_image, black_perc * 100)
    result = (norm_image - i_min) / (i_max - i_min)
    result[result > 1.0] = 1
    result[result < 0.0] = 0
    return result


# Box-filter

def box_filter(image, w, h):
    return image_utils.with_normalised(image, lambda image: __box_filter(image, w, h))

def __box_filter(image, w, h):
    # todo: throw if image is too small
    integral = image_utils.get_integral_image(image)

    result = np.zeros_like(image)

    for i in range(h, image.shape[0] - h):
        for k in range(w, image.shape[1] - w):
            window_size = (2 * h + 1) * (2 * w + 1)
            result[i, k] = image_utils.integral_get_sum(integral, i - h, k - w, i + h, k + w) / window_size

    return result[h : image.shape[0] - h - 1, w: image.shape[1] - w - 1]
