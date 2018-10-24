import numpy as np
from . import image_utils


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
            result[i, k] = image_utils.integral_get_sum(
                integral, i - h, k - w, i + h, k + w) / window_size

    return result[h: image.shape[0] - h - 1, w: image.shape[1] - w - 1]
