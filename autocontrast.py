from __future__ import print_function
from sys import argv
import os.path

import image_utils
import image_processing as imagic

def autocontrast(src_path, dst_path, white_perc, black_perc):
    image_utils.process_image(
        src_path,
        dst_path,
        lambda image: imagic.autocontrast(image, white_perc, black_perc)
    )


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] < 1
    assert 0 <= argv[4] < 1

    autocontrast(*argv[1:])
