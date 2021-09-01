"""
Find Maximum Likelihood Reconstruction
"""

import os
from warnings import resetwarnings
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from dival.measure import PSNR, SSIM
import matplotlib.pyplot as plt 

from dival.util.torch_losses import tv_loss
from odl.contrib.torch import OperatorModule


from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.reconstructors.iunet_reconstructor import IUNetReconstructor
from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule
#from cinn_for_imaging.datasets.lodopab_lpd.data_util import LoDoPaBLPDDataModule

model = "multi_scale" #"iUnet"

#%% configure the dataset
dataset = LoDoPaBDataModule(batch_size=1, impl='astra_cuda', sorted_by_patient=False)

dataset.prepare_data()
dataset.setup(stage='test')
ray_trafo = dataset.ray_trafo
num_test_images = 10# len(dataset.test_dataloader())

if model == 'iUnet':
    print("iUNet: \n")
    reconstructor = IUNetReconstructor(operator=ray_trafo, in_ch=1, img_size=None)

    #%% load the desired model checkpoint
    version = 'version_15' 
    chkp_name = 'epoch=24-step=37299'
    path_parts = ['..', 'experiments', 'lodopab', 'iunet', 'default', version, 'checkpoints', chkp_name + '.ckpt']
    chkp_path = os.path.join(*path_parts)
    reconstructor.load_learned_params(path=chkp_path, checkpoint=True)

else:
    print("Multi-Scale: \n")

    #%% initialize the reconstructor
    reconstructor = CINNReconstructor(
        ray_trafo=ray_trafo, 
        in_ch=1, 
        img_size=None)

    #%% load the desired model checkpoint
    version = 'version_7' 
    chkp_name = 'epoch=47-step=30671'
    path_parts = ['..', 'experiments', 'lodopab', 'multi_scale', 'default', version, 'checkpoints', chkp_name + '.ckpt']
    chkp_path = os.path.join(*path_parts)
    reconstructor.load_learned_params(path=chkp_path, checkpoint=True)

criterion = CINNNLLLoss(
    distribution=reconstructor.model.hparams.sample_distribution)

#%% move model to the GPU
reconstructor.samples_per_reco = 1000
reconstructor.max_samples_per_run = 500
#reconstructor.model.to('cuda')

print("Number of Params: ", sum(p.numel() for p in reconstructor.model.parameters() if p.requires_grad))
#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
pl.seed_everything(eval_seed)
reconstructor.model.eval()


op = OperatorModule(ray_trafo)

psnrs = [] 
ssims = []

psnrs_init = [] 
ssims_init = []
for i, batch in tqdm(zip(range(num_test_images),dataset.test_dataloader()), 
                        total=num_test_images):
    obs, gt = batch
    #obs = obs.to('cuda')
    #gt = gt.to('cuda')
    
    # create cond. input
    c = reconstructor.model.cond_net(obs)

    x_recs = [] 
    for i in range(4):
    # Init x_init as one sample from model T^{-1}(z,y)
        z_init = torch.randn(1, reconstructor.img_size[0]*reconstructor.img_size[1])#.to("cuda")
        x_init, _ = reconstructor.model.cinn(z_init, c=c, rev=True)

        x_rec = x_init[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]].detach().cpu().numpy()[0][0]

        x_recs.append(x_rec)


    fig, axes = plt.subplots(1,4, sharex=True, sharey=True)

    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x_recs[i], cmap="gray")
        ax.axis("off")
    plt.show()    
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
    ax1.imshow(gt.cpu()[0][0], cmap="gray")
    ax1.set_title("Groundtruth")

    ax2.imshow(x_rec.detach().cpu()[0][0], cmap="gray")
    ax2.set_title("Reconstruction. PNSR = " + str(psnrs[-1]))

    x_init = x_init[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]
    ax3.imshow(x_init.detach().cpu()[0][0], cmap="gray")
    ax3.set_title("Initialisierung T^{-1}(z;y). PNSR = " + str(PSNR(x_init.detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0])))

    plt.show()
    """

