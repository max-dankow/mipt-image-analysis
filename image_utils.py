import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image


def open_image(path):
    return np.asarray(Image.open(path))


def save_image(imagearray, path):
    Image.fromarray(imagearray).save(path)


def process_image(src_path, dst_path, action):
    source = open_image(src_path) if src_path is not None else get_example()
    result = action(source)
    if dst_path is None:
        show(result)
    else:
        save_image(result, dst_path)

def get_example():
    return scipy.misc.face()


def show(image, caption="image"):
    plt.figure(num=caption)
    plt.imshow(image, vmin=0, vmax=1)
    plt.show()


def with_normalised(image, action):
    if (np.any(image > 1)):
        # Type np.uint8 is very important, because otherwise colors would be incorrect
        # TODO: guarantee that values are in 0..255
        return (action(image / 255.0) * 255.0).astype(np.uint8)
    else:
        return action(image)


def get_integral_image(image):
    integral = np.zeros_like(image)

    integral[0, 0] = image[0, 0]
    for i in range(1, image.shape[0]):
        integral[i, 0] = image[i, 0] + integral[i - 1, 0]

    for i in range(1, image.shape[1]):
        integral[0, i] = image[0, i] + integral[0, i - 1]

    for i in range(1, image.shape[0]):
        for k in range(1, image.shape[1]):
            integral[i, k] = integral[i - 1, k] + integral[i, k - 1] - integral[i - 1, k - 1] + image[i, k]

    return integral


# Including ends
def integral_get_sum(integral, row_from, col_from, row_to, col_to):
    result = integral[row_to, col_to]
    if col_from <= 0 and row_from <= 0:
        return result

    if col_from > 0 and row_from > 0:
        result = result + integral[row_from - 1, col_from - 1]

    if col_from > 0:
        result = result - integral[row_to, col_from - 1]

    if row_from > 0:
        result = result - integral[row_from - 1, col_to]

    return result