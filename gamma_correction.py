from __future__ import print_function
from sys import argv
import os.path

from ImageProcessingCore.ImageProcessing import open_image_from


def gamma_correction(src_path, dst_path, a, b):
    open_image_from(src_path) \
        .gamma_correction(a, b) \
        .save_to(dst_path)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])
