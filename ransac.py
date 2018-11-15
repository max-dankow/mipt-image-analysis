from __future__ import print_function
from sys import argv
import os.path, json
import matplotlib.pyplot as plt
import numpy as np

def show_line(line, best_line, t, data):
    plt.scatter(data[:, 0], data[:, 1])

    dir_ = line[1] - line[0]
    ort = np.array([dir_[1], -dir_[0]])
    ort *= t / np.sum(ort**2)**0.5

    plt.plot(line[:, 0], line[:, 1], color='red')
    plt.plot((line + ort)[:, 0], (line + ort)[:, 1], color='yellow')
    plt.plot((line - ort)[:, 0], (line - ort)[:, 1], color='yellow')
    plt.plot(best_line[:, 0], best_line[:,1], color='green')
    plt.show()

def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    if abs(line_params[1]) < 1e-6:
        raise NotImplementedError

    h = img_size[0]
    w = img_size[1]

    # Generate inliers
    inliers_number = int(inlier_ratio * n_points)
    inliers_x = np.random.uniform(low=0, high=w, size=inliers_number)
    inliers_y = -(inliers_x * line_params[0] + line_params[2]) / line_params[1]
    inliers = np.stack((inliers_x, inliers_y), axis=1)
    center_y = inliers_y[inliers_number // 2]

    # Distort inliers with normal noise
    noise = np.random.normal(loc=0, scale=sigma, size=inliers_number)
    inliers[:, 1] += noise

    # Generate outliers
    outliers_number = n_points - inliers_number
    outliers = np.random.rand(outliers_number, 2)

    # Shift and scale outliers
    outliers[:, 0] *= w
    outliers[:, 1] = outliers[:, 1] * h - h // 2 + center_y

    # Concatenate and shuffle inliers and outliers together
    sample = np.concatenate((inliers, outliers))
    np.random.shuffle(sample)
    return sample


def compute_ransac_thresh(alpha, sigma):
    # Use Chebyshev's inequality:
    # Mean is 0, standard deviation is sigma**2
    # P(|delta| > threshold) <= (deviation) / (threshold**2)
    # and P should be less than 1 - alpha
    return (sigma**2 / (1 - alpha))**0.5


def compute_ransac_iter_count(conv_prob, inlier_ratio):
    n = np.log(1 - conv_prob) / np.log(1 - inlier_ratio**2)
    return int(n)


def compute_line_ransac(data, t, n):
    best_score = 0
    best_line = None
    for _ in range(n):
        a_ind = np.random.randint(0, len(data))
        b_ind = np.random.randint(0, len(data))

        if a_ind == b_ind:
            # todo: regenerate or ignore
            continue
        a = data[a_ind]
        b = data[b_ind]
        line_direction = b - a
        line_direction_norm = line_direction / np.sum(line_direction**2)**0.5
        score = 0
        # if best_line is not None:
        #     show_line(np.array([a, b]), best_line, t, data)

        inliers =[]
        for k, p in enumerate(data):
            if k == a_ind or k == b_ind:
                continue
            v = a - p
            dist = v[0] * line_direction_norm[1] - v[1] * line_direction_norm[0]
            if abs(dist) <= t:
                score += 1
                inliers.append(p)

        # inliers = np.array(inliers)
        # plt.scatter(data[:, 0], data[:, 1])
        # plt.scatter(inliers[:, 0], inliers[:, 1], color='green')
        # plt.show()
        if score > best_score:
            best_score = score
            best_line = np.array([a, b])

    best_a = best_line[0]
    best_b = best_line[1]
    # if best_a[0] > best_b[0]:
    #     best_a, best_b = best_b, best_a
    k = (best_b[1] - best_a[1]) / (best_b[0] - best_a[0])
    b = best_a[1] - best_a[0] * k 
    return k, b

def main():
    print(argv)
    assert len(argv) == 2
    assert os.path.exists(argv[1])

    with open(argv[1]) as fin:
        params = json.load(fin)

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used
    sigma - Gaussian noise
    alpha - probability of point is an inlier
    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """
    img_size = (params['w'], params['h'])
    line_params = (params['a'], params['b'], params['c'])
    data = generate_data(img_size,
                         line_params,
                         params['n_points'], params['sigma'],
                         params['inlier_ratio'])

    t = compute_ransac_thresh(params['alpha'], params['sigma'])
    n = compute_ransac_iter_count(params['conv_prob'], params['inlier_ratio'])

    print('threshold is', t)
    print('N iterations is', n)


    detected_line = compute_line_ransac(data, t, n)
    print(detected_line)

    plt.scatter(data[:, 0], data[:, 1], alpha=0.1)

    grid = np.linspace(0, img_size[0], num=2, endpoint=True)
    plt.plot(grid, grid * detected_line[0] + detected_line[1], color='red')
    plt.plot(grid, -(grid * line_params[0] + line_params[2]) / line_params[1], color='green')
    plt.show()


if __name__ == '__main__':
    main()
