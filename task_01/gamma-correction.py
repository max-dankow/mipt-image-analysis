import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def get_example():
    return scipy.misc.face()

def show(image, caption="image"):
    plt.figure(num=caption)
    plt.imshow(image, vmin=0, vmax=1)
    plt.show()

def gamma_correction(image, a, gamma):
    return __with_normalised(image, lambda image: __gamma_correction(image, a, gamma))

def __gamma_correction(norm_image, a, gamma):
    return a * (norm_image.copy() ** gamma)

def autocontrast_file(src_img_path, dst_img_path, white_perc, black_perc):
    raise NotImplementedError

def autocontrast(image, white_perc, black_perc):
    return __with_normalised(image, lambda image: __autocontrast(image, white_perc, black_perc))

def __autocontrast(norm_image, white_perc, black_perc):
    # Unfoturnately, method quantile() is not found in numpy
    i_max = np.percentile(norm_image, (1.0 - white_perc) * 100)
    i_min = np.percentile(norm_image, black_perc * 100)
    result = (norm_image - i_min) / (i_max - i_min)
    result[result > 1.0] = 1
    result[result < 0.0] = 0
    return result

def __with_normalised(image, action):
    if (np.any(image > 1)):
        # Type np.uint8 is very important, because otherwise colors would be incorrect
        # TODO: guarantee that values are in 0..255
        return (action(image / 255.0) * 255.0).astype(np.uint8)
    else:
        return action(image)

original = get_example()
contrasted = autocontrast(original, 0.3, 0.1)
show(contrasted, "Contrast")
#corrected = gamma_correction(original, 1, 2)

#for gamma in np.arange(0.25, 2., 0.25):
#    corrected = gamma_correction(original, 1, gamma)
#    show(corrected, "Gamma = %f" % gamma)