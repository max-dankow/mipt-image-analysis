import numpy as np
from .image_utils import compute_grey_hist


# Otsu


def otsu(grey_image):
    # todo: throw if not greyscale
    result = np.copy(grey_image)
    threshold = __compute_otsu_threshold(grey_image)
    result[result > threshold] = 255
    result[result <= threshold] = 0
    return result


def __compute_otsu_threshold(image):
    N = image.shape[0] * image.shape[1]
    hist = compute_grey_hist(image)

    total_weighted_count = np.dot(hist, np.arange(256))

    max_sigma = -1
    best_threshold = 0

    class_1_count = 0
    class_1_weighted_count = 0

    for t in range(256):
        class_1_count += hist[t]
        class_1_weighted_count += t * hist[t]

        if class_1_count == 0:
            continue

        if class_1_count == N:
            break

        class_1_probability = class_1_count / N
        class_1_mean = class_1_weighted_count / class_1_count

        class_2_probability = 1.0 - class_1_probability
        class_2_mean = (total_weighted_count - class_1_weighted_count) / (N - class_1_count)

        classes_mean_distance = class_1_mean - class_2_mean
        sigma = class_1_probability * class_2_probability * classes_mean_distance * classes_mean_distance

        if sigma > max_sigma:
            max_sigma = sigma
            best_threshold = t

    return best_threshold
