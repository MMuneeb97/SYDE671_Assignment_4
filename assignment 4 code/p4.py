import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    """
    Estimates the four possible R and T pairings from the Essential matrix E.
    
    Arguments:
        E - 3x3 Essential matrix
    Returns:
        RT_pairs - a list or structure containing the 4 possible [R | t] matrices
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation determinants
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    
    # The two possible Q matrices before enforcing det(Q) == +1
    Q1 = U @ W @ Vt
    Q2 = U @ W.T @ Vt
    
    # Ensure Q is a valid rotation matrix: R = (det Q) * Q
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2
    
    # The two possible translation vectors (u3 and -u3)
    t1 = U[:, 2:3]
    t2 = -U[:, 2:3]
    
    # Form the four possible [R | t] pairings
    RT_pairs = [
        np.hstack((R1, t1)),
        np.hstack((R1, t2)),
        np.hstack((R2, t1)),
        np.hstack((R2, t2))
    ]
    
    return RT_pairs

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    """
    Computes a linear estimate (DLT) of a 3D point given corresponding 
    2D image points and camera projection matrices.
    
    Arguments:
        image_points - list or array of 2D coordinates for the point in each image 
                       (can be inhomogeneous [x, y] or homogeneous [x, y, w])
        camera_matrices - list or array of 3x4 projection matrices [M1, M2, ...]
    Returns:
        P_3D - the 3D coordinates (x, y, z) of the triangulated point
    """
    A = []
    
    for pt, M in zip(image_points, camera_matrices):
        pt_flat = np.array(pt).flatten()
        # Extract inhomogeneous x and y coordinates
        if len(pt_flat) >= 3 and pt_flat[2] != 0:
            x = pt_flat[0] / pt_flat[2]
            y = pt_flat[1] / pt_flat[2]
        else:
            x, y = pt_flat[0], pt_flat[1]
            
        # Rows of the projection matrix M (3x4)
        m1 = M[0, :]
        m2 = M[1, :]
        m3 = M[2, :]
        
        # Form the two DLT equations per camera view
        row1 = x * m3 - m1
        row2 = y * m3 - m2
        
        A.append(row1)
        A.append(row2)
        
    A = np.array(A)
    
    # Solve A * P = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    P_hom = Vt[-1, :]  # Right singular vector corresponding to the smallest singular value
    
    # Convert from homogeneous coordinates (X, Y, Z, W) to 3D inhomogeneous (X/W, Y/W, Z/W)
    P_3D = P_hom[:3] / P_hom[3]
    
    return P_3D

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    """
    Computes the 2Mx1 reprojection error vector for a 3D point across M images.
    
    Arguments:
        point_3d - 3D point coordinates (X, Y, Z) or homogeneous (X, Y, Z, 1)
        image_points - measured points in each of the M images (Mx2 matrix)
        camera_matrices - camera projective matrices (Mx3x4 tensor)
    Returns:
        error - 2Mx1 reprojection error vector
    """
    M_count = camera_matrices.shape[0]
    errors = []
    
    # Ensure point_3d is homogeneous (4-element vector)
    if len(point_3d) == 3:
        P_hom = np.append(point_3d, 1.0)
    else:
        P_hom = point_3d

    for i in range(M_count):
        Mi = camera_matrices[i]
        pi = image_points[i]
        
        # Project 3D point onto camera i: y = M_i * P
        y = Mi @ P_hom
        
        # Inhomogeneous projection coordinates
        p_prime = y[:2] / y[2]
        
        # Residual error for image i: e_i = p_prime - p_i
        ei = p_prime - pi
        errors.append(ei)
        
    return np.hstack(errors)


def jacobian(point_3d, camera_matrices):
    """
    Computes the 2Mx3 Jacobian matrix of the reprojection error w.r.t the 3D point.
    
    Arguments:
        point_3d - 3D point coordinates (X, Y, Z)
        camera_matrices - camera projective matrices (Mx3x4 tensor)
    Returns:
        jacobian - 2Mx3 Jacobian matrix
    """
    M_count = camera_matrices.shape[0]
    J = []
    
    X, Y, Z = point_3d[0], point_3d[1], point_3d[2]
    P_hom = np.array([X, Y, Z, 1.0])

    for i in range(M_count):
        Mi = camera_matrices[i]
        
        # Rows of camera matrix Mi
        m1 = Mi[0, :]
        m2 = Mi[1, :]
        m3 = Mi[2, :]
        
        y = Mi @ P_hom
        y1, y2, y3 = y[0], y[1], y[2]
        
        # Derivatives using the quotient rule for p_i' = [y1/y3, y2/y3]^T w.r.t [X, Y, Z]
        # y3 * m_row - y_row * m3 for each coordinate
        j_x = (y3 * m1[:3] - y1 * m3[:3]) / (y3 ** 2)
        j_y = (y3 * m2[:3] - y2 * m3[:3]) / (y3 ** 2)
        
        J.append(j_x)
        J.append(j_y)
        
    return np.vstack(J)

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    """
    Estimates an optimized 3D point from multi-view correspondences by first 
    computing a linear DLT estimate and refining it via Gauss-Newton 
    non-linear optimization.
    
    Arguments:
        image_points - measured points in each of the M images (Mx2 matrix)
        camera_matrices - camera projective matrices (Mx3x4 tensor)
    Returns:
        P_refined - the refined 3D coordinates (x, y, z)
    """
    # 1. Get initial estimate using linear DLT triangulation
    initial_guess = linear_estimate_3d_point(image_points, camera_matrices)
    P = np.array(initial_guess, dtype=np.float64)
    
    # 2. Run Gauss-Newton optimization for 10 iterations
    num_iterations = 10
    for _ in range(num_iterations):
        e = reprojection_error(P, image_points, camera_matrices)
        J = jacobian(P, camera_matrices)
        
        jt_j = J.T @ J
        jt_e = J.T @ e
        
        # Solve (J^T * J) * delta = J^T * e via least squares for numerical stability
        delta, _, _, _ = np.linalg.lstsq(jt_j, jt_e, rcond=None)
        
        # Update estimate
        P = P - delta.flatten()
        
    return P

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    """
    Distinguishes the correct R, T pair from the Essential matrix E 
    using the cheirality constraint (positive depth in both camera frames).
    """
    # Robustly handle image_points whether passed as [pts1, pts2] or a single matched array
    if isinstance(image_points, (list, tuple)) and len(image_points) == 2:
        pts1, pts2 = image_points[0], image_points[1]
    else:
        # If image_points contains all matches, split them or handle accordingly 
        # (e.g., if shape is Nx4 where first 2 cols are pts1 and last 2 cols are pts2)
        pts1 = image_points[:, 0, :]
        pts2 = image_points[:, 0, :]
        
    N = pts1.shape[0]
    
    # Get the 4 possible [R | t] pairs for the second camera
    RT_candidates = estimate_initial_RT(E)
    
    # First camera is fixed at the origin: P1 = K * [I | 0]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    best_RT = None
    max_positive_depths = -1
    
    for RT2 in RT_candidates:
        P2 = K @ RT2
        camera_matrices = [P1, P2]
        
        positive_depth_count = 0
        
        for i in range(N):
            pt_pair = [pts1[i], pts2[i]]
            
            # Triangulate 3D point in the first camera's frame
            P_3d = linear_estimate_3d_point(pt_pair, camera_matrices)
            P_3d_hom = np.append(P_3d, 1.0)
            
            # Depth in Camera 1 frame
            depth1 = P_3d[2]
            
            # Transform point to Camera 2 frame and check depth
            P_cam2 = RT2 @ P_3d_hom
            depth2 = P_cam2[2]
            
            if depth1 > 0 and depth2 > 0:
                positive_depth_count += 1
                
        if positive_depth_count > max_positive_depths:
            max_positive_depths = positive_depth_count
            best_RT = RT2
            
    return best_RT

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
