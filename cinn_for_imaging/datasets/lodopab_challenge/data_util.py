# -*- coding: utf-8 -*-
import h5py
import numpy as np

def read_h5_file(filename, out=None, n=None):
    """
    Read images from HDF5 file into an array.

    Parameters
    ----------
    filename : str
        Filename (path) of the HDF5 file.
    out : array, optional
        Array in which to store the read data.
        If `None`, a new array is created.
    n : int, optional
        Number of images to read.
        If `None`, it is inferred from `out` if provided or from the file
        otherwise.

    Raises
    ------
    ValueError
        If the dataset in the file does not have three dimensions or if
        `n` is too large.

    Returns
    -------
    out : array
        Array holding the read data.
    """
    with h5py.File(filename, 'r') as file:
        shape = file['data'].shape
    if len(shape) != 3:
        raise ValueError('expected dataset of shape (n, im_x, im_y)')
    if n is None:
        n = out.shape[0] if out is not None else shape[0]
    if shape[0] < n:
        raise ValueError('requested {:d} images, but axis 0 of dataset only '
                         'has length {:d}'.format(n, shape[0]))
    if out is not None and out.shape[0] < n:
        raise ValueError('requested {:d} images, but axis 0 of `out` only has '
                         'length {:d}'.format(n, out.shape[0]))
    if out is None:
        out = np.zeros((n,) + shape[1:], dtype=np.float32)
    with h5py.File(filename, 'r') as file:
        file['data'].read_direct(out, np.s_[:n], np.s_[:n])
    return out
