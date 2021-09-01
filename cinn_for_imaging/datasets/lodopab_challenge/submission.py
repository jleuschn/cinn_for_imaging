# -*- coding: utf-8 -*-
"""
Functions for creating submissions.

To create a submission, the easiest way is to call :func:`save_reconstruction`
for each reconstruction computed from the challenge set observations.
Afterwards the written files can be zipped (e.g. using :func:`pack_submission`)
and uploaded to the challenge website.

*Note:* A submission must contain reconstructions for the whole challenge set.

The submission format is as follows:

A submission is a zip file containing several HDF5 files.
Each HDF5 file contains 128 reconstructions in form of a dataset named
``'data'`` of shape ``(128,) + IMAGE_SHAPE``.
The HDF5 filenames must end with indices (before the extension dot),
e.g. ``'reco_000.hdf5'`` or ``'0.h5'`` are valid filenames.
These indices, which must be consecutive starting from 0 or 1, determine the
order of the files.
The dataset in the last HDF5 file may have a smaller first dimension, but also
otherwise any spare part will be ignored (this note addresses the likely case
that the challenge set size is not a multiple of ``128``).
"""
import os
import re
from math import ceil
import h5py
import numpy as np
from zipfile import ZipFile

NUM_SAMPLES_PER_FILE = 128

def validate_and_get_submission_filenames(path):
    """
    Check that the HDF5 files in `path` form a valid submission.

    path : str
        Directory containing the files for submission.
        If `path` contains a single directory, it is used instead of `path`.

    Returns
    -------
    filenames : list of str
        List of paths of the submission files.

    Raises
    ------
    RuntimeError
        If the submission is invalid.
    """
    input_filenames = os.listdir(path)
    if (len(input_filenames) == 1
            and os.path.isdir(os.path.join(path, input_filenames[0]))):
        # enter single top-level directory
        path = os.path.join(path, input_filenames[0])
        input_filenames = os.listdir(path)
    filenames = [f for f in input_filenames
                 if os.path.isfile(os.path.join(path, f))
                 and os.path.splitext(f)[1] in ['.hdf5', '.h5', '.hdf']]
    indices = []
    for file in filenames:
        match = re.search(r'\d+$', os.path.splitext(file)[0])
        if match is None:
            raise RuntimeError('expected index at end of HDF filename, e.g. '
                               '"reco_01.hdf5", but found "{}"'.format(file))
        ind = int(match.group(0))
        if ind in indices:
            raise RuntimeError('HDF5 file indices must be unique, found '
                               'double index {:d}'.format(ind))
        indices.append(ind)
    if not (min(indices) in [0, 1] and np.all(np.diff(sorted(indices)) == 1)):
        raise RuntimeError('HDF5 file indices must be either 0- or 1-based '
                           'consecutive indices')
    filenames = [os.path.join(path, f) for i, f in
                 sorted(zip(indices, filenames), key=lambda p: p[0])]
    return filenames

def pack_submission(path, filename=None, **kwargs):
    """
    Zip-pack the files from a submission directory.
    Checks the files before.

    Parameters
    ----------
    path : str
        Directory containing the HDF5 files for submission.
    filename : str, optional
        Filename (path) for the zip file.
        If `None`, the zip file is created next to the directory `path` with
        the same name extended by ``'.zip'``.

    Additional parameters
    ---------------------
    kwargs : dict
        Keyword arguments passed to :func:`zipfile.ZipFile`.
    """
    submission_filenames = validate_and_get_submission_filenames(path)
    if filename is None:
        filename = os.path.normpath(path) + '.zip'
    with ZipFile(filename, 'w', **kwargs) as zf:
        for file in submission_filenames:
            zf.write(file)

def save_reconstruction(path, idx, reconstruction, filename_basis='reco'):
    """
    Save a reconstruction for submission (in a HDF5 file).

    If the file does not exists, it is created and filled with NaN for the
    other reconstructions (each file contains 128 reconstructions).
    The filename is
    ``os.path.join(path, filename_basis + '_{:03d}.hdf5'.format(idx//128))``
    (for example ``/some/path/reco_002.hdf5``).

    Parameters
    ----------
    path : str
        Path to the directory in which the reconstruction will be saved.
    idx : int
        Index of the sample, i.e. a number in ``range(NUM_IMAGES)``.
    reconstruction : array-like
        Reconstruction.
    filename_basis : str, optional
        Basis of the HDF5 filename.
        Default is ``'reco'``.
    """
    reconstruction = np.ascontiguousarray(reconstruction)
    file_idx = idx // NUM_SAMPLES_PER_FILE
    image_idx = idx % NUM_SAMPLES_PER_FILE
    filename = os.path.join(
        path, filename_basis + '_{:03d}.hdf5'.format(file_idx))
    shape = reconstruction.shape
    with h5py.File(filename, 'a') as file:
        dataset = file.require_dataset(
            'data', shape=(NUM_SAMPLES_PER_FILE,) + shape,
            dtype=np.float32, exact=True, fillvalue=np.nan, chunks=True)
        dataset[image_idx] = reconstruction

def save_reconstructions(path, key, reconstructions, filename_basis='reco'):
    """
    Save multiple reconstructions for submission (in HDF5 file(s)).

    If any file does not exists, it is created and filled with NaN for the
    other reconstructions (each file contains 128 reconstructions).
    The filename is
    ``os.path.join(path, filename_basis + '_{:03d}.hdf5'.format(idx//128))``
    (for example ``/some/path/reco_002.hdf5``).

    Parameters
    ----------
    path : str
        Path to the directory in which the reconstruction will be saved.
    key : range
        Indices of the samples, i.e. a range within ``range(NUM_IMAGES)``.
        The step must be positive.
    reconstructions : array-like
        Reconstructions, shape ``(n,) + IMAGE_SHAPE``.
    filename_basis : str, optional
        Basis of the HDF5 filenames.
        Default is ``'reco'``.
    """
    if not isinstance(key, range):
        raise TypeError('`key` expected to have type `range`')
    range_ = key
    if range_.step < 0:
        raise ValueError('key {} invalid (negative step)'.format(key))
    if range_[0] < 0:
        raise IndexError("key {} invalid (negative start value)".format(key))
    reconstructions = np.ascontiguousarray(reconstructions)
    shape = reconstructions.shape[1:]
    range_files = range(range_[0] // NUM_SAMPLES_PER_FILE,
                        range_[-1] // NUM_SAMPLES_PER_FILE + 1)
    # compute slice objects
    slices_files = []
    slices_data = []
    data_count = 0
    for i in range_files:
        if i == range_files.start:
            start = range_.start % NUM_SAMPLES_PER_FILE
        else:
            start = (range_.start - i*NUM_SAMPLES_PER_FILE) % range_.step
        if i == range_files[-1]:
            stop = range_[-1] % NUM_SAMPLES_PER_FILE + 1
        else:
            __next_start = ((range_.start - (i+1)*NUM_SAMPLES_PER_FILE)
                            % range_.step)
            stop = (__next_start - range_.step) % NUM_SAMPLES_PER_FILE + 1
        s = slice(start, stop, range_.step)
        slices_files.append(s)
        len_slice = ceil((s.stop-s.start) / s.step)
        slices_data.append(slice(data_count, data_count+len_slice))
        data_count += len_slice
    # write files
    for i, slc_f, slc_d in zip(range_files, slices_files, slices_data):
        filename = os.path.join(
            path, filename_basis + '_{:03d}.hdf5'.format(i))
        with h5py.File(filename, 'a') as file:
            dataset = file.require_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + shape,
                dtype=np.float32, exact=True, fillvalue=np.nan, chunks=True)
            dataset.write_direct(reconstructions, slc_d, slc_f)
