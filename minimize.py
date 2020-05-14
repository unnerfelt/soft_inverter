import argparse
import h5py
import numpy as np
from load_image import load_image
from misc import create_matrix

# Argument parsing.
parser = argparse.ArgumentParser(description='Find fitness as function of C and p')
parser.add_argument('-i', '--image', metavar='I', required=True, type=str, help='input images')
parser.add_argument('-g', '--greens', metavar='G', required=True, type=str, nargs='+', help='input greens')
parser.add_argument('-o', '--output', metavar='O', required=True, type=str, help='output')
parser.add_argument('-f', '--frames', metavar='N', required=True, type=int, nargs='+', help='input frames')
parser.add_argument('-s', '--subpixels', metavar='XxYxWxH', required=True, type=str, help='Subpixels')

args = parser.parse_args()

frames, times, tfilter, filter_dark, images = load_image(args.image, args.frames, args.subpixels)

# Calculate the fitness of each C in C_space, as well as all p from the greens function.
def fitness(C_space, green, images, theta_grid):
    n_p = green.shape[1]
    result = np.zeros((len(C_space), n_p))
    count = 0
    for p_idx in range(n_p):
        for idx, C in enumerate(C_space):
            mat = create_matrix(p_idx, C, green, theta_grid)
            total_res = 0
            for image in images:
                count += 1
                flat_image = np.ndarray.flatten(image)
                x, res, _, _ = np.linalg.lstsq(mat, flat_image, rcond=1e-4)
                b = mat.dot(x)
                msq = np.sum((flat_image - b) ** 2)
                # total_res += res
                total_res += msq


            result[idx, p_idx] = total_res
    return result

# Set C_space to iterate through.
# Logarithmic seems to fit the best
C_space = np.logspace(np.log10(1), np.log10(400), 100)

# Iterate through green's functions.
results = []
p_space = []
count = 0
for green_path in args.greens:
    print("Progress: {}%".format(count / len(args.greens) * 100))
    count += 1
    with h5py.File(green_path, 'r') as f:
        func = f['func'][:]
        theta_grid = f['param2'][:]
        res = fitness(C_space, func, images, theta_grid)
        results.append(res)
        p_space.append(f['param1'][:])

p_space = np.concatenate(p_space)
results = np.concatenate(results, axis=1)

with h5py.File(args.output, 'w') as f:
    f['fitness'] = results
    f['C_space'] = C_space
    f['p_space'] = p_space
    f['C_min'] = C_space[np.argmin(np.min(results, axis=1))]
    f['p_min'] = p_space[np.argmin(np.min(results, axis=0))]
