
#Question2
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
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height, width = im.shape[0], im.shape[1]
    
    T = np.array([
        [1, 0, -width / 2.0],
        [0, 1, -height / 2.0],
        [0, 0, 1]
    ])
    
    e_translated = T @ e
    e1, e2 = e_translated[0], e_translated[1]
    
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
    
    def to_inhomogeneous(pts):
        if pts.shape[1] == 3:
            return pts[:, :2] / pts[:, 2:3]
        return pts[:, :2]
        
    pts1_inh = to_inhomogeneous(points1)
    pts2_inh = to_inhomogeneous(points2)
    N = pts1_inh.shape[0]
    
    pts1_hom = np.column_stack([pts1_inh, np.ones(N)])
    pts2_hom = np.column_stack([pts2_inh, np.ones(N)])
    
    p_hat = (H2 @ pts2_hom.T).T
    p_hat_inh = p_hat[:, :2] / p_hat[:, 2:3]
    
    W =  np.column_stack([pts1_inh, np.ones(N)])
    
    # Let's solve Wa = b using least squares
    # W has shape (N, 3), b has shape (N, 1) or (N,) -> using x-coordinate of p_hat_inh
    b = p_hat_inh[:, 0]
    
    a, _, _, _ = np.linalg.lstsq(W, b, rcond=None)
    a1, a2, a3 = a[0], a[1], a[2]
    
    H_A = np.array([
        [a1, a2, a3],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    H1 = H_A @ H2 @ M
    
    return H1, H2



if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
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
