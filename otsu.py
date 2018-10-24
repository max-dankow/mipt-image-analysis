from __future__ import print_function
from sys import argv
import os.path

from ImageProcessingCore.ImageProcessing import open_image_from


def otsu(src_path, dst_path):
    open_image_from(src_path) \
        .convert_to_greyscaled() \
        .binarize_greyscaled_with_otsu() \
        .save_to(dst_path)


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
