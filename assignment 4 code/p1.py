import numpy as np
import matplotlib.pyplot as plt
from epipolar_utils import *
from skimage.io import imread

def lls_eight_point_alg(points1, points2):
    N = points1.shape[0]

    if points1.shape[1] == 2:
        x1, y1 = points1[:, 0], points1[:, 1]
    else:
        x1 = points1[:, 0] / points1[:, 2]
        y1 = points1[:, 1] / points1[:, 2]

    if points2.shape[1] == 2:
        x2, y2 = points2[:, 0], points2[:, 1]
    else: 
        x2 = points2[:, 0] / points2[:, 2]
        y2 = points2[:, 1] / points2[:, 2]

    A = np.column_stack([
        x2 * x1,
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        np.ones(N)
    ])

    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F = f.reshape(3, 3)
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[-1] = 0.0
    F_refined = U_f @ np.diag(S_f) @ Vt_f

    return F_refined


def normalized_eight_point_alg(points1, points2):
    def to_inhomogenous(pts):
        if pts.shape[1] == 3:
            return pts[:, :2] / pts[:, 2:3]
        return pts[:, :2]

    pts1_inh = to_inhomogenous(points1)
    pts2_inh = to_inhomogenous(points2)

    N = pts1_inh.shape[0]

    mean1 = np.mean(pts1_inh, axis=0)
    max_dist1 = np.max(np.sqrt(np.sum((pts1_inh - mean1)**2, axis=1)))
    scale1 = np.sqrt(2) / max_dist1 if max_dist1 > 0 else 1.0

    T1 = np.array([
        [scale1, 0, -scale1 * mean1[0]],
        [0, scale1, -scale1 * mean1[1]],
        [0, 0, 1]
    ])

    pts1_hom = np.column_stack([pts1_inh, np.ones(N)])
    pts1_norm = (T1 @ pts1_hom.T).T

    mean2 = np.mean(pts2_inh, axis=0)
    max_dist2 = np.max(np.sqrt(np.sum((pts2_inh - mean2)**2, axis=1)))
    scale2 = np.sqrt(2) / max_dist2 if max_dist2 > 0 else 1.0

    T2 = np.array([
        [scale2, 0, -scale2 * mean2[0]],
        [0, scale2, -scale2 * mean2[1]],
        [0, 0, 1]
    ])

    pts2_hom = np.column_stack([pts2_inh, np.ones(N)])
    pts2_norm = (T2 @ pts2_hom.T).T

    x1_n, y1_n = pts1_norm[:, 0], pts1_norm[:, 1]
    x2_n, y2_n = pts2_norm[:, 0], pts2_norm[:, 1]

    A = np.column_stack([
        x2_n * x1_n,
        x2_n * y1_n,
        x2_n,
        y2_n * x1_n,
        y2_n * y1_n,
        y2_n,
        x1_n,
        y1_n,
        np.ones(N)
    ])

    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F_norm = f.reshape(3, 3)

    U_f, S_f, Vt_f = np.linalg.svd(F_norm)
    S_f[-1] = 0.0
    F_norm_refined = U_f @ np.diag(S_f) @ Vt_f

    F = T2.T @ F_norm_refined @ T1

    return F


def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a, b, c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c - a * x) / b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x, y, _ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

  
    fig = plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')


def compute_distance_to_epipolar_lines(points1, points2, F):
    if points1.shape[1] == 2:
        pts1_hom = np.column_stack([points1, np.ones(points1.shape[0])])
    else:
        pts1_hom = points1

    if points2.shape[1] == 2:
        pts2_hom = np.column_stack([points2, np.ones(points2.shape[0])])
    else:
        pts2_hom = points2

    lines = (F.T @ pts2_hom.T).T

    numerator = np.abs(np.sum(pts1_hom * lines, axis=1))
    denominator = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
    
    # Corrected distance calculation
    distances = numerator / denominator
    average_distance = np.mean(distances)

    return average_distance


if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", 
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", 
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", 
            F_normalized)
        print("Distance to lines in image 1 for normalized:", 
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", 
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()