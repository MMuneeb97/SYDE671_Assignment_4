import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
'''
def factorization_method(points_im1, points_im2):
    def to_inhomogeneous(pts):
        if pts.shape[1] == 3:
            return pts[:, :2] / pts[:, 2:3]
        return pts[:, :2]

    p1 = to_inhomogeneous(points_im1)
    p2 = to_inhomogeneous(points_im2)

    # Center each image's points at their centroid
    p1_centered = p1 - np.mean(p1, axis=0)
    p2_centered = p2 - np.mean(p2, axis=0)

    # Measurement matrix: 4 x N
    W = np.vstack([p1_centered.T, p2_centered.T])

    # Economy SVD avoids building an N x N matrix
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    U_3 = U[:, :3]
    sqrt_S3 = np.diag(np.sqrt(S[:3]))
    Vt_3 = Vt[:3, :]

    motion = U_3 @ sqrt_S3        # 4 x 3
    structure = sqrt_S3 @ Vt_3    # 3 x N

    return structure, motion


if __name__ == '__main__':
    base_path = os.path.join(script_dir, 'data')
    datasets = ['set1', 'set1_subset']

    for dataset_name in datasets:
        im_set = os.path.join(base_path, dataset_name)
        if not os.path.exists(im_set):
            print(f"Skipping {dataset_name}: folder not found.")
            continue

        print(f"Running factorization on: {dataset_name}")

        points_im1 = get_data_from_txt_file(os.path.join(im_set, 'pt_2D_1.txt'))
        points_im2 = get_data_from_txt_file(os.path.join(im_set, 'pt_2D_2.txt'))

        gt_path = os.path.join(im_set, 'pt_3D.txt')
        points_3d = get_data_from_txt_file(gt_path) if os.path.exists(gt_path) else None

        assert points_im1.shape == points_im2.shape

        structure, motion = factorization_method(points_im1, points_im2)

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0, :], structure[1, :], structure[2, :], ax1)
        ax1.set_title(f'Factorization Method ({dataset_name})')

        if points_3d is not None:
            ax2 = fig.add_subplot(122, projection='3d')
            scatter_3D_axis_equal(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], ax2)
            ax2.set_title('Ground Truth')

        plt.show()