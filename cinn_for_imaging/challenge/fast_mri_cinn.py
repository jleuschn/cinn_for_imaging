"""
Submission CINNReconstructor on the 'fastmri' test set.
"""

import os

from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl

from cinn_for_imaging.reconstructors.mri_reconstructor import CINNReconstructor
from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule
from cinn_for_imaging.datasets.fast_mri.submission import save_reconstructions, convert_fnames_to_v2

#%% configure the dataset
dataset = FastMRIDataModule(batch_size=1, num_data_loader_workers=8)
dataset.prepare_data()
dataset.setup()
num_test_images = len(dataset.test_dataloader())

#%% initialize the reconstructor
reconstructor = CINNReconstructor(
    in_ch=1, 
    img_size=(320, 320),
    max_samples_per_run=100)
#%% load the desired model checkpoint
version = 'version_3' 
chkp_name = 'epoch=472-step=342451'
path_parts = ['..', 'experiments', 'fast_mri', 'cinn', 'default', 'from_magneto', version, 'checkpoints', chkp_name + '.ckpt']

submission_path_parts = path_parts[:-2]
submission_path_parts.append('submission')
submission_path = os.path.join(*submission_path_parts)
Path(submission_path).mkdir(parents=True, exist_ok=True)

submission_path = Path(submission_path)


chkp_path = os.path.join(*path_parts)
reconstructor.load_learned_params(path=chkp_path, checkpoint=True)
#reconstructor.init_model()
#reconstructor.model.init_params()
#%% move model to the GPU
reconstructor.model.to('cuda')

#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
reconstructor.samples_per_reco = 100
reconstructor.max_samples_per_run = 100
pl.seed_everything(eval_seed)
reconstructor.model.eval()

reconstructions = defaultdict(list)
with torch.no_grad():
    for i, batch in tqdm(zip(range(num_test_images),dataset.test_dataloader()), 
                         total=num_test_images):
        obs, _, mean, std, filename, slice_num, _ = batch               
        obs = obs.to('cuda')
        
        #print(filename, slice_num)

        # create reconstruction from observation
        reco, reco_std = reconstructor._reconstruct(obs,return_std=True)
        reco = np.asarray(reco)
        reco = reco * std.numpy() + mean.numpy() 
        reco = np.clip(reco, a_min=0, a_max=0.005) # max value in train data: 0.0024660237
        #print(reco.shape)
        #import matplotlib.pyplot as plt 
        #plt.figure()
        #plt.imshow(reco, cmap="gray")
        #plt.show()

        reconstructions[filename[0]].append((int(slice_num[0]), reco))




# save outputs
for fname in reconstructions:
    reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])
save_reconstructions(reconstructions, submission_path / "reconstructions")