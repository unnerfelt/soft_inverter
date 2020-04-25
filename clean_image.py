import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import pyqtgraph.console
import pyqtgraph as pg
import numpy.linalg
import h5py
import scipy as sp
import scipy.ndimage
import argparse

parser = argparse.ArgumentParser(description='Filter noise from synchrotron images')
parser.add_argument('input', metavar='INPUT', type=str, help='input path, hdf5 format')
parser.add_argument('output', metavar='OUTPUT', type=str, help='output path, hdf5 format')
parser.add_argument('-t', '--threshold', metavar='X', type=float, default=0.9, help='set threshold, default 0.9')

args = parser.parse_args()

with h5py.File(args.input, 'r') as vf:
    frames = vf['frames'][:].transpose((0, 2, 1)).astype(np.double) # Flip image
    times = vf['times'][:]

def filter_frames(frames, threshold):
    mean = np.mean(frames, axis=0)
    mean_gauss = sp.ndimage.filters.gaussian_filter(mean, 2)

    filter_dark = (mean < mean_gauss * 0.9) | (mean < np.max(mean) * 0.2)
    
    smoothed = sp.ndimage.gaussian_filter(mean, 0.8)

    frames = frames * smoothed / mean

    time_median = np.copy(frames)
    for x in range(400):
        for y in range(400):
            time_median[:, x, y] = sp.ndimage.median_filter(frames[:, x, y], size=3)

    tfilter = np.abs(time_median - frames) > 144

    tfilter_forward = np.abs(time_median[:-1, :, :] / frames[1:, :, :])
    tfilter_backward = np.abs(time_median[1:, :, :] / frames[:-1, :, :])
    #tfilter_forward = frames[1:, :, :] - time_median[:-1, :, :]
    #tfilter_backward = frames[:-1, :, :] - time_median[1:, :, :]

    tfilter[1:-1] = (tfilter_forward[:-1] < threshold) & (tfilter_backward[1:] < threshold)
    #tfilter[1:-1] = (tfilter_forward[:-1] > threshold) & (tfilter_backward[1:] > threshold)

    frames_nn = np.copy(frames)
    for x in range(400):
        for y in range(400):
            tslice = np.copy(frames[:, x, y])
            tslice_filter = tfilter[:, x, y]

            tslice[tslice_filter] = np.interp(times[tslice_filter], times[~tslice_filter], tslice[~tslice_filter])
            frames_nn[:, x, y] = tslice
        
    return frames_nn, tfilter, filter_dark

frames, tfilter, filter_dark = filter_frames(frames, args.threshold)

with h5py.File(args.output, 'w') as vf:
    vf.create_dataset('times', data=times)
    vf.create_dataset('frames', data=frames)
    vf.create_dataset('tfilter', data=tfilter)
    vf.create_dataset('filter_dark', data=filter_dark)
