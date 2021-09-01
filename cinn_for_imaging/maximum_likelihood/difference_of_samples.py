"""
Find Maximum Likelihood Reconstruction
"""

import os
from warnings import resetwarnings
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from dival.measure import PSNR, SSIM
import matplotlib.pyplot as plt 
from dcp.utils.losses import tv_loss

from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule


#%% configure the dataset
#dataset = LoDoPaBDataModule(batch_size=1, impl='astra_cuda',
#                            sorted_by_patient=False,
#                            num_data_loader_workers=8)
dataset = LoDoPaBDataModule(batch_size=1, impl='astra_cuda')

dataset.prepare_data()
dataset.setup(stage='test')
ray_trafo = dataset.ray_trafo
num_test_images = len(dataset.test_dataloader())

#%% initialize the reconstructor
reconstructor = CINNReconstructor(
    ray_trafo=ray_trafo, 
    in_ch=1, 
    img_size=None)

#%% load the desired model checkpoint
#experiment_name = 'lodopab_lpd'
#version = 'version_4' 
#chkp_name = 'epoch=19'
#path_parts = ['..', 'experiments', experiment_name, 'default',
#              version, 'checkpoints', chkp_name + '.ckpt']
#chkp_path = os.path.join(*path_parts)
version = 'version_4' 
chkp_name = 'epoch=47-step=286559'
path_parts = ['..', 'experiments', 'lodopab', 'default', version, 'checkpoints', chkp_name + '.ckpt']
#path_parts = ['cinn_for_imaging', 'experiments', 'gaussian_normal', 'lightning_logs',
#              version, 'checkpoints', chkp_name + '.ckpt']
chkp_path = os.path.join(*path_parts)
reconstructor.load_learned_params(path=chkp_path, checkpoint=True)

criterion = CINNNLLLoss(
    distribution=reconstructor.model.hparams.sample_distribution)

#%% move model to the GPU
reconstructor.samples_per_reco = 1000
reconstructor.max_samples_per_run = 500
reconstructor.model.to('cuda')

print("Number of Params: ", sum(p.numel() for p in reconstructor.model.parameters() if p.requires_grad))
#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
pl.seed_everything(eval_seed)
reconstructor.model.eval()




for i, batch in tqdm(zip(range(num_test_images),dataset.test_dataloader()), 
                        total=num_test_images):
    obs, gt = batch
    obs = obs.to('cuda')
    gt = gt.to('cuda')
    
    c = reconstructor.model.cond_net(obs)

    z1 = torch.randn(1, reconstructor.img_size[0]*reconstructor.img_size[1]).to("cuda")
    z2 = -z1


    x1, _ = reconstructor.model.cinn(z1, c=c, rev=True)
    x1 = x1[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]

    x2, _ = reconstructor.model.cinn(z2, c=c, rev=True)
    x2 = x2[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,8), sharex=True, sharey=True)

    ax1.imshow(x1.detach().cpu()[0][0], cmap="gray")
    ax1.set_title("z1")
    ax2.imshow(x2.detach().cpu()[0][0], cmap="gray")
    ax2.set_title("z2")

    ax3.imshow(torch.abs(x1.detach().cpu()[0][0]-x2.detach().cpu()[0][0]), cmap="gray")
    ax3.set_title("|x1 - x2|")



    plt.show()


    break