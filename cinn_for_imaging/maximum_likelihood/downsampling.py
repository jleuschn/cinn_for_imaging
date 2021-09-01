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

    z = torch.randn(1, reconstructor.img_size[0]*reconstructor.img_size[1]).to("cuda")


    x_outs, _ = reconstructor.model.cinn(z, c=c, rev=True, intermediate_outputs=True)
    #x = x[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]

    #for i, out in enumerate(x_outs):
    #    print(i, out[0], out[1])



    for idx, key in enumerate(x_outs.keys()):
        print(idx, key, x_outs[key].shape)

        if idx == 92:

            fig, axes = plt.subplots(2,2)
            fig.suptitle(key)
            for i, ax in enumerate(axes.ravel()):

                ax.imshow(x_outs[key][0,i,:,:].detach().cpu().numpy(), cmap="gray")

            plt.show()

        if idx == 112:

            fig, ax = plt.subplots(1,1)
            fig.suptitle(key)
            ax.imshow(x_outs[key][0,0,:,:].detach().cpu().numpy(), cmap="gray")

            plt.show()
    #print(x_outs[112])    
    break