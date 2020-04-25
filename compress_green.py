import numpy as np
import argparse
import h5py

parser = argparse.ArgumentParser(description='Linearly compress or combine SOFT Green\'s functions.')
parser.add_argument('greens', metavar='GREEN', type=str, nargs='+', help='green functions')
parser.add_argument('output', metavar='OUTPUT', type=str, help='output path, hdf5 format')
parser.add_argument('-r', '--radial', metavar='N', required=True, type=int, help='set radial resolution')

args = parser.parse_args()

#print(args.greens)

def row(i, n, m):
    x = (m - 1) / (n - 1)
    a = np.zeros(n)
    if i == (m - 1):
        a[n - 1] = 1
    else:
        j = i / x
        a[int(j)] = 1 - j % 1
        a[int(j) + 1] = j % 1
    return a

def interp_mat(n, m):
    rows = [row(i, n, m) for i in range(m)]
    return np.stack(rows, axis=0)

def interp_greens(data, n):
    m = data.shape[0]
    rsize = np.prod(data.shape[1:])
    nsize = (n,) + data.shape[1:]
    mat = np.transpose(interp_mat(n, m))
    return mat.dot(data.reshape(m, rsize)).reshape(nsize)

with h5py.File(args.output, 'w') as out:
    param1 = []
    total_func = []
    for green in args.greens:
        with h5py.File(green, 'r') as f:
            for name, data in f.items():
                if name in ['func', 'param1']:
                    continue
                if name in out:
                    if not np.array_equal(out[name][:], data[:]):
                        print(out[name][:])
                        print(data[:])
                        assert False
                else:
                    out[name] = data[:]
            data = f['func'][:]
            data = interp_greens(data, args.radial)
            total_func.append(data)
            param1.append(f['param1'][:])
    out['func'] = np.concatenate(total_func, axis=1)
    out['param1'] = np.concatenate(param1)
