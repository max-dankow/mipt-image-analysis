from __future__ import print_function
from sys import argv
import os.path

from ImageProcessingCore.ImageProcessing import open_image_from


def box_flter(src_path, dst_path, w, h):
    open_image_from(src_path) \
        .box_filter(w, h) \
        .save_to(dst_path)


if __name__ == '__main__':
    assert len(argv) == 5
    #assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])
