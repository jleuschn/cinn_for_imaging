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

model = "iUnet"


def pm_loss(x):
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(torch.log(1 + (dh[..., :-1, :] + dw[..., :, :-1])**2))


#%% configure the dataset
dataset = LoDoPaBDataModule(batch_size=1, impl='astra_cuda', sorted_by_patient=False)

dataset.prepare_data()
dataset.setup(stage='test')
ray_trafo = dataset.ray_trafo
num_test_images = 100# len(dataset.test_dataloader())

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
reconstructor.model.to('cuda')

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
    obs = obs.to('cuda')
    gt = gt.to('cuda')
    
    # create cond. input
    c = reconstructor.model.cond_net(obs)

    # Init: x_init as zero
    #x_init = torch.zeros(1,1,reconstructor.img_size[0], reconstructor.img_size[0]).to("cuda")

    # Init x_init as fbp 
    #x_init = reconstructor.model.cond_net.fbp_layer(obs)
    #x_init = reconstructor.model.cond_net.img_padding(x_init)  

    # Init x_init as one sample from model T^{-1}(z,y)
    z_init = torch.randn(1, reconstructor.img_size[0]*reconstructor.img_size[1]).to("cuda")
    x_init, _ = reconstructor.model.cinn(z_init, c=c, rev=True)

    # Wenn wir nur PM oder TV in der Optimierung benutzen, können wir schon hier x_init croppen. 
    # Für die Auswertung der log-likelihood brauchen wir jedoch die ganze Größe 
    # Im ersten Schritt des cINN haben wir das padding. Darauf muss man aufpassen, wenn man PSNR oder SSIM auswertet 
    # oder TV, PM berechnet
    #x_init = x_init[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]


    x_rec = torch.nn.Parameter(x_init.clone(), requires_grad=True)
    #optim = torch.optim.SGD([x_rec], lr=3e-3)
    optim = torch.optim.Adam([x_rec], lr=1e-4)

    gamma = 10.
    psnsrs_temp = [] 
    ssims_temp = []
    mse_temp = [] 
    nll_temp = []
    for _ in range(100):
        optim.zero_grad()

        zz, log_jac = reconstructor.model.cinn(x_rec, c=c, rev=False)
        ndim_total = zz.shape[-1]
        nll = torch.mean(zz**2) / 2 - torch.mean(log_jac) / ndim_total
        mse = torch.sum((op(x_rec[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]) - obs)**2)

        #tv = tv_loss(x_rec) 
        #loss = mse + gamma*nll#+ gamma*tv
        #mse =  torch.sum((op(x_rec) - obs)**2)
        #r = pm_loss(x_rec)#torch.sum(x_rec)**2#tv_loss(x_rec)#
        
        #print(i, mse.item(), nll.item())
        loss = mse + gamma*nll

        loss.backward(retain_graph=True)
        #print(i, mse.item(), nll.item())
    
        optim.step()

        with torch.no_grad():
            x_rec[x_rec < 0] = 0.

        psnsrs_temp.append(PSNR(x_rec[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]].detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0]))
        ssims_temp.append(SSIM(x_rec[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]].detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0]))
        mse_temp.append(mse.item())
        nll_temp.append(nll.item())
    
    """
    fig, axes = plt.subplots(2,2)

    axes[0,0].plot(psnsrs_temp)
    axes[0,0].set_title("PSNR")
    axes[0,1].plot(ssims_temp)
    axes[0,1].set_title("SSIM")

    axes[1,0].plot(mse_temp)
    axes[1,0].set_title("MSE")

    axes[1,1].plot(nll_temp)
    axes[1,1].set_title("NLL")
    plt.show()
    """

    x_rec = x_rec[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]

    psnrs.append(PSNR(x_rec.detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0]))
    ssims.append(SSIM(x_rec.detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0]))
    #print(psnrs[-1], ssims[-1])
    print(np.mean(psnrs), np.mean(ssims))
 
    x_init = x_init[:,:,:reconstructor.model.op.domain.shape[0],:reconstructor.model.op.domain.shape[1]]
    psnrs_init.append(PSNR(x_init.detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0]))
    ssims_init.append(SSIM(x_init.detach().cpu().numpy()[0][0], gt.cpu().numpy()[0][0]))


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


mean_psnr = np.mean(psnrs)
std_psnr = np.std(psnrs)
mean_ssim = np.mean(ssims)
std_ssim = np.std(ssims)


print('---')
print(gamma)
print('Results:')
print('mean psnr: {:f}'.format(mean_psnr))
print('std psnr: {:f}'.format(std_psnr))
print('mean ssim: {:f}'.format(mean_ssim))
print('std ssim: {:f}'.format(std_ssim))
print('Results of Initialization:')
print('mean psnr: {:f}'.format(np.mean(psnrs_init)))
print('std psnr: {:f}'.format(np.std(psnrs_init)))
print('mean ssim: {:f}'.format(np.mean(ssims_init)))
print('std ssim: {:f}'.format(np.std(ssims_init)))