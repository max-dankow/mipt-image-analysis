from .image_utils import open_image, show, save_image

from .gamma_correction import gamma_correction
from .autocontrast import autocontrast
from .box_filter import box_filter
from .otsu import otsu
from .grey_scale import convert_to_greyscale

class ImageProcessing:
    def __init__(self, image):
        self.image = image

    def then_apply(self, image_processing_action):
        self.image = image_processing_action(self.image)
        return self

    def show(self):
        show(self.image)
        return self

    def save_to(self, path):
        save_image(self.image, path)

    # Extention methods

    def gamma_correction(self, a, gamma):
        self.image = gamma_correction(self.image, a, gamma)
        return self


    def autocontrast(self, white_perc, black_perc):
        self.image = autocontrast(self.image, white_perc, black_perc)
        return self

    def box_filter(self, w, h):
        self.image = box_filter(self.image, w, h)
        return self


    def convert_to_greyscaled(self):
        self.image = convert_to_greyscale(self.image)
        return self


    def binarize_greyscaled_with_otsu(self):
        self.image = otsu(self.image)
        return self


def open_image_from(path):
    image = open_image(path)
    return ImageProcessing(image)
