import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def get_example():
    return scipy.misc.face()

def show(image):
    plt.imshow(image, vmin=0, vmax=1)
    plt.show()

def gamma_correction(image, a, gamma):
    #print(type(image[0,0,0]))
    if (np.any(image > 1)):
        # Type np.uint8 is very important, because otherwise colors would be incorrect
        # TODO: guarantee that values are in 0..255
        return (__gamma_correction(image / 255.0, a, gamma) * 255.0).astype(np.uint8)
        
    else:
        return __gamma_correction(image, a, gamma)


def __gamma_correction(image, a, gamma):
    return a * (image.copy() ** gamma)

original = get_example()
corrected = gamma_correction(original, 1, 2)

for gamma in np.arange(0, 2, 0.25):
    print("Gamma = %f" % gamma)
    corrected = gamma_correction(original, 1, gamma)
    show(corrected)