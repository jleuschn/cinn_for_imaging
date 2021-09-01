# -*- coding: utf-8 -*-
from dival import get_standard_dataset, get_reference_reconstructor
from dival.datasets.dataset import Dataset
from dival.datasets.cached_dataset import generate_cache_files, CachedDataset

class LoDoPaBLearnedPDDataset(Dataset):
    def __init__(self, impl='astra_cuda'):
        self.dataset = get_standard_dataset('lodopab', impl=impl)
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.random_access = True
        self.num_elements_per_sample = 2
        self.reconstructor = get_reference_reconstructor(
            'learnedpd', 'lodopab', impl=impl)
        self.shape = (self.dataset.shape[1], self.dataset.shape[1])
        super().__init__(space=(self.dataset.space[1], self.dataset.space[1]))

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        (out_obs, out_gt) = out
        out_basis = (out_obs is not False, out_gt)
        obs_basis, gt = self.dataset.get_sample(index, part=part,
                                                out=out_basis)
        if isinstance(out_obs, bool):
            obs = (self.reconstructor.reconstruct(obs_basis)
                   if out_obs else None)
        else:
            self.reconstructor.reconstruct(obs_basis, out=out_obs)
            obs = out_obs
        return (obs, gt)

if __name__ == '__main__':
    IMPL = 'astra_cuda'
    dataset = LoDoPaBLearnedPDDataset(impl=IMPL)

    CACHE_FILES = {
        'train': (
            '/localdata/dival_dataset_caches/'  # TODO adapt
            'cache_train_lodopab_learnedpd.npy',
            None),
        'validation': (
            '/localdata/dival_dataset_caches/'  # TODO adapt
            'cache_validation_lodopab_learnedpd.npy',
            None)}

    # generate_cache_files(dataset, CACHE_FILES)

    cached_dataset = CachedDataset(dataset, dataset.space, CACHE_FILES)

    for i in range(3):
        obs, gt = dataset.get_sample(i, part='train')
        obs2, gt2 = dataset.get_sample(i, part='train')
        obs_cached, gt_cached = cached_dataset.get_sample(i, part='train')
        import numpy as np
        print(obs_cached.shape, gt_cached.shape)
        # print(np.max(np.abs(obs-obs2)))
        # print(np.max(np.abs(obs-obs_cached)))
        assert np.allclose(np.asarray(obs), np.asarray(obs2))
        assert np.allclose(np.asarray(gt), np.asarray(gt2))
        assert np.allclose(np.asarray(obs), np.asarray(obs_cached), atol=1e-7)
        assert np.allclose(np.asarray(gt), np.asarray(gt_cached), atol=1e-7)
