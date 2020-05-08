import argparse
import h5py
import numpy as np
from load_image import load_image
from misc import create_matrix

import pdb
import pyqtgraph as pg

# Argument parsing.
parser = argparse.ArgumentParser(description='Find fitness as function of C and p')
parser.add_argument('-i', '--image', metavar='I', required=True, type=str, help='input images')
parser.add_argument('-g', '--greens', metavar='G', required=True, type=str, nargs='+', help='input greens')
parser.add_argument('-o', '--output', metavar='O', required=True, type=str, help='output')
parser.add_argument('-f', '--frames', metavar='N', required=True, type=int, nargs='+', help='input frames')
parser.add_argument('-s', '--subpixels', metavar='XxYxWxH', required=True, type=str, help='Subpixels')
parser.add_argument('-M', '--matrix', metavar='M', default='identity', type=str, help='Matrix to use in Tikhonov regularization. Values: identity (default), diff')

parser.add_argument('-p', metavar='P', required=True, type=float, help='Momentum')
parser.add_argument('-C', metavar='C', required=True, type=float, help='C parameter')

args = parser.parse_args()

frames, times, tfilter, filter_dark, images = load_image(args.image, args.frames, args.subpixels)

# A bit of an ugly solution to get the greens function with the closest momentum.
minimum_path = ''
smallest_p = None
for green_path in args.greens:
    with h5py.File(green_path, 'r') as f:
        p_min = np.min(np.abs(f['param1'][:] - args.p))
        if not smallest_p or p_min < smallest_p:
            smallest_p = p_min
            minimum_path = green_path

with h5py.File(minimum_path, 'r') as f:
    p_idx = np.argmin(np.abs(f['param1'][:] - args.p))
    func = f['func'][:]
    r_grid = f['r'][:]
    theta_grid = f['param2'][:]
    mat = create_matrix(p_idx, args.C, func, theta_grid)

# Do the Tikhonov regularization and similar things.
#x, res, _, _ = np.linalg.lstsq(mat, np.ndarray.flatten(images[0]), rcond=None)
u, s, vt = np.linalg.svd(mat, full_matrices=False)

def invert_tikhonov(alpha, u, s, vt, images):
    s = np.copy(s)
    f = np.square(s) / (np.square(s) + np.square(alpha))
    s = np.divide(1, s, where=(s>0))
    s = s * f
    pinv = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))

    xs = []
    imgs = []
    msq = 0

    for image in images:
        flat_image = np.ndarray.flatten(image)
        x = pinv.dot(flat_image)
        b = mat.dot(x)
        msq += np.sum((flat_image - b) ** 2)
        
        xs.append(x)
        imgs.append(b.reshape(80, 80)) # TODO: Fix this.
    return xs, imgs, msq

def invert_tikhonov_diff(alpha, mat, images):
    M = (np.eye(200) - np.eye(200, k=1))[:-1]
    mat2 = np.vstack([mat, M * alpha])

    xs = []
    imgs = []
    msq = 0

    for image in images:
        flat_image = np.ndarray.flatten(image)
        target = np.hstack([flat_image, np.zeros(M.shape[0])])
        x, res, _, _ = np.linalg.lstsq(mat2, target, rcond=None)
        b = mat.dot(x)

        msq = np.sum((flat_image - b) ** 2)
        xs.append(x)
        imgs.append(b.reshape(80, 80)) # TODO: Fix this.

        msq += msq

    return xs, imgs, msq

# Now we try to find alpha as large as possible without sacrificing fitness.
# This is done through a simple binary search.

if args.matrix == "diff":
    inv_func = lambda alpha: invert_tikhonov_diff(alpha, mat, images)
else:
    inv_func = lambda alpha: invert_tikhonov(alpha, u, s, vt, images)

def evaluate(alpha):
    xs, imgs, msq = inv_func(alpha)
    return msq

upper = 100
lower = -100

minimum = evaluate(10.0 ** lower)
maximum = evaluate(10.0 ** upper)

tol = 1e-4
tol_it = 0.1
def is_good(alpha):
    val = (evaluate(alpha) - minimum) / (maximum - minimum)
    return val < tol

while (upper - lower) > tol_it:
    mid = (upper + lower) / 2
    if is_good(10.0 ** mid):
        lower = mid
    else:
        upper = mid

xs, imgs, msq = inv_func(10.0 ** lower)

# Save the results
with h5py.File(args.output, 'w') as f:
    for x, img, target, frame_idx in zip(xs, imgs, images, args.frames):
        f['recreated_' + str(frame_idx)] = img
        f['target_' + str(frame_idx)] = target
        f['radial_' + str(frame_idx)] = x
    f['p'] = args.p
    f['C'] = args.C
    f['r_grid'] = r_grid
