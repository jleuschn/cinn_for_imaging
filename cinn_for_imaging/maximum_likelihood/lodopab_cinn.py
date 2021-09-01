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
from dival.util.plot import plot_images
import matplotlib.pyplot as plt 
from dcp.utils.losses import tv_loss

from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule
#from cinn_for_imaging.datasets.lodopab_lpd.data_util import LoDoPaBLPDDataModule


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

    reco, reco_std = reconstructor._reconstruct(obs,return_std=True)

    plt.figure()
    plt.imshow(reco_std/(reco+ 1e-4))
    plt.show()

    """
    print("PSNR of mean: ", PSNR(reco, gt.cpu().numpy()[0][0]))
    print("SSIM of mean: ", SSIM(reco, gt.cpu().numpy()[0][0]))
    # x_init = 0
    #x_init = torch.zeros(1,1,reconstructor.img_size[0], reconstructor.img_size[0]).to("cuda")

    # x_init = fbp 
    #x_init = reconstructor.model.cond_net.fbp_layer(obs)
    #x_init = reconstructor.model.cond_net.img_padding(x_init)  

    # x_init = T^{-1}(z,y)
    z_init = torch.randn(1, reconstructor.img_size[0]*reconstructor.img_size[1]).to("cuda")
    x_init, _ = reconstructor.model.cinn(z_init, c=c, rev=True)

    print(gt.shape, x_init.shape)
    x_rec = torch.nn.Parameter(x_init.clone(), requires_grad=True)
    optim = torch.optim.Adam([x_rec], lr=1e-3)

    gamma = 1e-5

    psnrs = [] 
    ssims = []
    tvs = [] 
    nlls = []
    for i in range(1000):
        optim.zero_grad()

        zz, log_jac = reconstructor.model.cinn(x_rec, c=c, rev=False)
        ndim_total = zz.shape[-1]

        nll = torch.mean(zz**2) / 2 - torch.mean(log_jac) / ndim_total
        tv = tv_loss(x_rec) 

        loss = nll + gamma*tv

        loss.backward(retain_graph=True)
        print(i, nll.item(), tv.item())
    
        optim.step()

        with torch.no_grad():
            x_rec[x_rec < 0] = 0.

        psnrs.append(PSNR(x_rec.detach().cpu().numpy()[0,0,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]], gt.cpu().numpy()[0][0]))
        ssims.append(SSIM(x_rec.detach().cpu().numpy()[0,0,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]], gt.cpu().numpy()[0][0]))
        tvs.append(tv.item())
        nlls.append(nll.item())
        if (i+1) % 100 == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            fig.suptitle("Likelihood {} at iteration {}".format(nll.item(), i))
            ax1.imshow(gt.cpu()[0][0], cmap="gray")
            ax1.set_title("Groundtruth")

            ax2.imshow(x_rec.detach().cpu()[0,0,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]], cmap="gray")
            ax2.set_title("Reconstruction")

            ax3.imshow(zz.detach().cpu().reshape(reconstructor.img_size[0], reconstructor.img_size[1]))
            ax3.set_title("z")

            plt.savefig("maximum_likelihood/reconstruction_{}.png".format(i))
            plt.show()
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(gt.cpu()[0][0], cmap="gray")
    ax1.set_title("Groundtruth")

    ax2.imshow(x_rec.detach().cpu()[0,0,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]], cmap="gray")
    ax2.set_title("Reconstruction")

    ax3.imshow(zz.detach().cpu().reshape(reconstructor.img_size[0], reconstructor.img_size[1]))
    ax3.set_title("z")

    plt.savefig("maximum_likelihood/reconstruction.png")
    plt.show()


    fig, axes = plt.subplots(2,2)
    fig.suptitle("Initialisierung x == T^{-1}(z;y)")
    axes[0,0].plot(psnrs)
    axes[0,0].set_title("PSNR")
    axes[0,0].set_xlabel("Iteration")
    axes[0,0].set_ylabel("PSNR")

    axes[0,1].plot(ssims)
    axes[0,1].set_title("SSIM")
    axes[0,1].set_xlabel("Iteration")
    axes[0,1].set_ylabel("SSIM")

    axes[1,0].plot(nlls)
    axes[1,0].set_title("NLL")
    axes[1,0].set_xlabel("Iteration")
    axes[1,0].set_ylabel("- log p(x|y)")

    axes[1,1].plot(tvs)
    axes[1,1].set_title("TV")
    axes[1,1].set_xlabel("Iteration")
    axes[1,1].set_ylabel("TV(x)")

    plt.show()
    break 
    """