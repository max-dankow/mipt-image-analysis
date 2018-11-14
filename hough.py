from __future__ import print_function
from sys import argv
import cv2
import numpy as np
import ImageProcessingCore.image_utils as utils


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude

import math
def hough_transform(img, theta, rho):
    img = img / np.max(img)
    h = img.shape[0]
    w = img.shape[1]

    # Generate grid
    # todo: rhos[-1] may be less than max_rho
    max_rho = (h**2 + w**2) ** 0.5
    rhos = np.arange(0, max_rho, step=rho)
    thetas = np.arange(0, 2 * math.pi, theta)
    grid = np.zeros(shape=(len(rhos), len(thetas)))

    for y in range(h):
        for x in range(w):
            for k, th in enumerate(thetas):
                r = x * math.cos(th) + y * math.sin(th)
                if r < 0 or r > rhos[-1]:
                    continue

                grid[math.floor(r / rho), k] += img[y, x]

    return grid, thetas, rhos

def to_k_b_line(rho, theta):
    if abs(math.sin(theta)) < 1e-6:
        # raise NotImplementedError
        return 1, 1

    return -1./math.tan(theta), rho / math.sin(theta)


def get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta):
    top = np.dstack(np.unravel_index(np.argsort(-ht_map.ravel()), (len(rhos), len(thetas))))[0]
    #top = np.argsort(ht_map)[:n_lines]
    result = []
    latest_result = None
    ttop = np.zeros_like(ht_map)
    for i in range(20):
        ttop[top[i][0], top[i][1]] = 1.0
    cv2.imshow('top', ttop)
    cv2.waitKey(0)

    for k, x in enumerate(top[:10]):
        # todo: make mean with the next theta
        print(x)
        r = rhos[x[0]]
        th = thetas[x[1]]
        print(x[1], x[0])
        print(r, th)
        k, b = to_k_b_line(r, th)
        print('y=',k, '*x + ', b)
        #if latest_result is None or latest_result.
    # todo: merge similar
    return 

if __name__ == '__main__':
    assert len(argv) == 9

    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None
    # cv2.imshow('demo', image)
    # cv2.waitKey(0)

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    print(ht_map)
    ht_map /= np.max(ht_map)
    cv2.imshow('ht_map', ht_map)
    cv2.waitKey(0)
    # cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta)
    # with open(dst_lines_path, 'w') as fout:
    #     for line in lines:
    #         fout.write('%0.3f, %0.3f\n' % line)
