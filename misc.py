import numpy as np

# Generate matrix for inverting.
def create_matrix(p_idx, C, green, theta_grid):
    ntheta = green.shape[2]
    nr = green.shape[0]
    ipixels = green.shape[3]
    jpixels = green.shape[4]

    mat = green[:, p_idx, :, :, :]
    theta_func = np.exp(C * np.cos(theta_grid)) / np.exp(C) * C
    mat = np.reshape(np.transpose(mat, axes=(1, 0, 2, 3)), (ntheta, nr * ipixels * jpixels))
    mat = np.transpose(np.reshape(theta_func.dot(mat), (nr, ipixels * jpixels)))
    
    return mat
