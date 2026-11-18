import os
import sys

# Make sure local modules (p1, epipolar_utils) can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1, :]

    if abs(e[2]) > 1e-10:
        e = e / e[2]
    else:
        e = e / np.linalg.norm(e)

    return e

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height, width = im.shape[0], im.shape[1]

    # Translate image center to the origin
    T = np.array([
        [1, 0, -width / 2.0],
        [0, 1, -height / 2.0],
        [0, 0, 1]
    ])

    e_translated = T @ e
    e1, e2 = e_translated[0], e_translated[1]

    # Rotate epipole onto the x-axis
    alpha = 1 if e1 >= 0 else -1
    norm_e = np.sqrt(e1**2 + e2**2)

    if norm_e > 1e-10:
        c1 = alpha * (e1 / norm_e)
        c2 = alpha * (e2 / norm_e)
    else:
        c1, c2 = 1.0, 0.0

    R = np.array([
        [c1, c2, 0],
        [-c2, c1, 0],
        [0, 0, 1]
    ])

    e_rotated = R @ e_translated
    f = e_rotated[0]

    # Send the epipole to infinity
    if abs(f) > 1e-10:
        G = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1.0 / f, 0, 1]
        ])
    else:
        G = np.eye(3)

    T_inv = np.linalg.inv(T)
    H = T_inv @ G @ R @ T

    return H

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)

    e_x = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])

    v_T = np.array([1, 1, 1])
    M = e_x @ F + np.outer(e2, v_T)

    def to_homogeneous(pts):
        if pts.shape[1] == 3:
            pts = pts[:, :2] / pts[:, 2:3]
        else:
            pts = pts[:, :2]
        return np.column_stack([pts, np.ones(pts.shape[0])])

    pts1_hom = to_homogeneous(points1)
    pts2_hom = to_homogeneous(points2)

    # H0 maps image 1 into the rectified frame (up to the affine H_A)
    H0 = H2 @ M

    # FIX: fit H_A on the points AFTER H0 / H2 are applied
    p1_hat = (H0 @ pts1_hom.T).T
    p1_hat = p1_hat / p1_hat[:, 2:3]

    p2_hat = (H2 @ pts2_hom.T).T
    p2_hat = p2_hat / p2_hat[:, 2:3]

    # Solve  p1_hat @ a = x-coordinates of p2_hat  (least squares)
    a, _, _, _ = np.linalg.lstsq(p1_hat, p2_hat[:, 0], rcond=None)

    H_A = np.array([
        [a[0], a[1], a[2]],
        [0,    1,    0],
        [0,    0,    1]
    ])

    H1 = H_A @ H0

    return H1, H2


# Estimate the size of the warped image so we can catch blow-ups early
def warped_bounds(H, im):
    h, w = im.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    c = H @ corners
    c = c[:2] / c[2]
    return c.min(axis=1), c.max(axis=1)


if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set + '/image1.jpg')
    im2 = imread(im_set + '/image2.jpg')
    points1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)

    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)

    # Sanity check: how big will the warped images be?
    for name, H, im in [("im1", H1, im1), ("im2", H2, im2)]:
        lo, hi = warped_bounds(H, im)
        print(f"{name} warped bounds: min={lo}, max={hi}")
        if np.max(hi - lo) > 10000:
            raise RuntimeError(
                f"{name} warp is huge ({hi - lo}); the homography is likely wrong "
                "and compute_rectified_image would take forever."
            )
    print('')

    # Transforming the points by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2, :]
    new_points2 /= new_points2[2, :]
    new_points1 = new_points1.T
    new_points2 = new_points2.T

    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)

    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()