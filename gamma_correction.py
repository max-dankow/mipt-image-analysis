from __future__ import print_function
from sys import argv
import os.path

import image_utils
import image_processing as imagic


def gamma_correction(src_path, dst_path, a, b):
    image_utils.process_image(
        src_path,
        dst_path,
        lambda image: imagic.gamma_correction(image, a, b)
    )


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])
