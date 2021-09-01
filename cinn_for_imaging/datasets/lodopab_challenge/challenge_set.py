# -*- coding: utf-8 -*-
"""
Functions for accessing the observation data of the LoDoPaB-CT challenge.

You probably need to adjust ``config['data_path']``.
This can be done by importing the ``config`` variable from this module and
setting its ``'data_path'`` value, e.g.:

.. code-block:: python

    from lodopab_challenge.challenge_set import config
    config['data_path'] = '/path/to/challenge_set'

Alternatively, you can directly edit the code of this module.
"""
import os
from math import ceil
import numpy as np
from odl import uniform_discr
from odl.tomo import parallel_beam_geometry
import h5py

# adjust this path to the location of the challenge set
config = {'data_path': '/localdata/lodopab_challenge_set'}  # TODO adapt

NUM_SAMPLES_PER_FILE = 128
NUM_IMAGES = 3678
IMAGE_SHAPE = (362, 362)
NUM_ANGLES = 1000
NUM_DET_PIXELS = 513
PHOTONS_PER_PIXEL = 4096
ORIG_MIN_PHOTON_COUNT = 0.1
MU_WATER = 20  # M^-1
MU_AIR = 0.02  # M^-1
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]

image_domain = uniform_discr(MIN_PT, MAX_PT, IMAGE_SHAPE, dtype=np.float32)
geometry = parallel_beam_geometry(
    image_domain, num_angles=NUM_ANGLES, det_shape=(NUM_DET_PIXELS,))
obs_domain = uniform_discr(geometry.partition.min_pt,
                           geometry.partition.max_pt,
                           (NUM_ANGLES, NUM_DET_PIXELS), dtype=np.float32)

def get_observation(idx, out=None):
    """
    Return an observation sample from the lodopab challenge set.

    Parameters
    ----------
    idx : int
        Index of the sample, i.e. a number in ``range(3678)`` (or in
        ``range(-3678, 0)``, then ``3678`` is added).
    out : odl element or array, optional
        Array in which to store the observation.
        Must have shape ``(1000, 513)``.
        If `None`, a new odl element is created.

    Raises
    ------
    IndexError
        If `idx` is out of bounds.

    Returns
    -------
    out : odl element or array
        Array holding the observation.
    """
    if idx >= NUM_IMAGES or idx < -NUM_IMAGES:
        raise IndexError("index {} out of bounds ({:d})"
                         .format(idx, NUM_IMAGES))
    if idx < 0:
        idx += NUM_IMAGES
    file_index = idx // NUM_SAMPLES_PER_FILE
    index_in_file = idx % NUM_SAMPLES_PER_FILE
    if out is None:
        out = obs_domain.zero()
    with h5py.File(os.path.join(config['data_path'],
                                'observation_challenge_{:03d}.hdf5'
                                .format(file_index)), 'r') as file:
        file['data'].read_direct(np.asarray(out)[np.newaxis],
                                 np.s_[index_in_file:index_in_file+1],
                                 np.s_[0:1])
    return out

def get_observations(key, out=None):
    """
    Return observation samples from the lodopab challenge set.

    Parameters
    ----------
    key : slice or range
        The indices of the samples.
        Only positive steps are supported.
        If a range is passed, all elements must be in ``range(3678)``.
    out : array, optional
        Array in which to store the observations.
        Must have shape ``(n, 1000, 513)``.
        If `None`, a new array is created.

    Raises
    ------
    TypeError
        If `key` does not have type `slice` or `range`.
    ValueError
        If `key` has negative step.
    IndexError
        If `key` is out of bounds.

    Returns
    -------
    out : array
        Array holding the observations.
        Shape: ``(n, 1000, 513)``.
    """
    if isinstance(key, slice):
        key_start = (0 if key.start is None else
                        (key.start if key.start >= 0 else
                        max(0, NUM_IMAGES+key.start)))
        key_stop = (NUM_IMAGES if key.stop is None else
                    (key.stop if key.stop >= 0 else
                        max(0, NUM_IMAGES+key.stop)))
        range_ = range(key_start, key_stop, key.step or 1)
    elif isinstance(key, range):
        range_ = key
    else:
        raise TypeError('`key` expected to have type `slice` or `range`')
    if range_.step < 0:
        raise ValueError('key {} invalid, negative steps are not '
                         'implemented yet'.format(key))
    if range_[0] < 0 or range_[-1] >= NUM_IMAGES:
        raise IndexError("key {} out of bounds ({:d})"
                         .format(key, NUM_IMAGES))
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
    # read data
    if out is None:
        out = np.empty((len(range_), NUM_ANGLES, NUM_DET_PIXELS),
                       dtype=np.float32)
    for i, slc_f, slc_d in zip(range_files, slices_files, slices_data):
        with h5py.File(os.path.join(config['data_path'],
                                    'observation_challenge_{:03d}.hdf5'
                                    .format(i)), 'r') as file:
            file['data'].read_direct(out, slc_f, slc_d)
    return out

def generator():
    """Yield observation samples from the lodopab challenge set.

    The generator stops after yielding all ``3678`` observations.

    Yields
    ------
    observation : odl element
        Array holding the observation.
        Shape: ``(1000, 513)``.
    """
    num_files = ceil(NUM_IMAGES / NUM_SAMPLES_PER_FILE)
    for i in range(num_files):
        with h5py.File(os.path.join(config['data_path'],
                                    'observation_challenge_{:03d}.hdf5'
                                    .format(i)), 'r') as file:
            observation_data = (
                file['data'][:] if i < num_files-1 else
                file['data'][:NUM_IMAGES-(num_files-1)*NUM_SAMPLES_PER_FILE])
        for obs_arr in observation_data:
            observation = obs_domain.element(obs_arr)
            yield observation

def transform_to_pre_log(obs, inplace=True):
    """
    Transform observation(s) to pre-log scale.

    The returned `obs` relates to the corresponding ground truth `gt` via
    ``obs = exp(-ray_trafo(gt * MU_MAX)) + noise``, where
    ``MU_MAX = 81.35858 = MU(3071 HU)`` is the factor, by which the ground
    truth image is normalized.

    *Note:* After calling this function one may call
    ``replace_min_photon_count(obs, 0., obs_is_post_log=False)`` in order to
    revert the replacement of zero photon counts that was applied for the
    post-log data.
    
    Parameters
    ----------
    obs : odl element or array
        Observation(s) in post-log scale.
    inplace : bool, optional
        Whether to write the result directly to `obs`.
        Default: ``True``.

    Returns
    -------
    obs : odl element or array
        Observation(s) in pre-log scale.
    """
    if inplace:
        obs *= MU_MAX
    else:
        obs = obs * MU_MAX
    np.exp(-obs, out=obs)
    return obs

def replace_min_photon_count(obs, min_photon_count,
                             obs_is_post_log=True, inplace=True):
    """
    Replace the values for zero photons in observation(s), occuring in
    directions of high attenuation.
    The value originally used for the dataset is ``0.1``.
    This function replaces those values using a different min. photon count.

    Parameters
    ----------
    obs : odl element or array
        Observation(s).
        If `obs_is_post_log` is ``False``, `obs` must be in pre-log scale,
        otherwise `obs` must be in post-log scale (the default case).
    min_photon_count : float
        Minimum photon count used as a replacement for zero photons.
        The value originally used for the dataset is ``0.1``.
    obs_is_post_log : bool, optional
        Whether `obs` is in post-log scale.
        Default: ``True``.
    inplace : bool, optional
        Whether to write the result directly to `obs`.
        Default: ``True``.

    Returns
    -------
    obs : odl element or array
        Observation with replaced minimum photon count.
    """
    mask = np.empty(obs.shape, dtype=np.bool)
    if obs_is_post_log:
        thres0 = 0.5 * (
            -np.log(ORIG_MIN_PHOTON_COUNT / PHOTONS_PER_PIXEL)
            - np.log(1. / PHOTONS_PER_PIXEL)) / MU_MAX
        np.greater_equal(obs, thres0, out=mask)
        replacement_value = (-np.log(min_photon_count / PHOTONS_PER_PIXEL)
                             / MU_MAX)
    else:
        thres0 = 0.5 * (ORIG_MIN_PHOTON_COUNT + 1.) / PHOTONS_PER_PIXEL
        np.less(obs, thres0, out=mask)
        replacement_value = min_photon_count / PHOTONS_PER_PIXEL
    if inplace:
        obs[mask] = replacement_value
    else:
        obs = np.where(mask, replacement_value, obs)
    return obs
