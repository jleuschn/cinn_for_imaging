"""
Benchmark IUNetReconstructor on different test set.
"""

import os

from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from dival.measure import PSNR, SSIM
from dival.util.plot import plot_images
from matplotlib.pyplot import savefig

from cinn_for_imaging.reconstructors.iunet_reconstructor import IUNetReconstructor
from cinn_for_imaging.util.torch_losses import CINNNLLLoss


#%% test configuration
dataset_name = 'lodopab'  # 'lodopab', 'fast_mri', 'ellipses'
model_version = 'version_14' 
chkp_name = 'epoch=147-step=165611'
server = 'dummy'  # TODO insert server name
num_test_images = -1  # choose -1 to use all
samples_per_reco = 1000

#%% configure the dataset
if dataset_name == 'lodopab':
    from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule
    dataset = LoDoPaBDataModule(batch_size=1, impl='astra_cuda',
                                sorted_by_patient=False,
                                num_data_loader_workers=0)
    dataset.prepare_data()
    dataset.setup(stage='test')
    dataloader = dataset.test_dataloader

elif dataset_name == 'ellipses':
    from cinn_for_imaging.datasets.ellipses.data_util import EllipsesDataModule
    dataset = EllipsesDataModule(batch_size=1, impl='astra_cuda', 
                                 num_data_loader_workers=0)
    dataset.prepare_data()
    dataset.setup(stage='test')
    dataloader = dataset.test_dataloader

elif dataset_name == 'fast_mri':
    from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule
    dataset = FastMRIDataModule(batch_size=1,
                                num_data_loader_workers=0)
    dataset.prepare_data()
    dataset.setup()
    dataloader = dataset.val_dataloader

if dataset_name == 'fast_mri':
    op = dataset.operator
else:
    op = dataset.ray_trafo
    
data_range = dataset.data_range

if num_test_images == -1:
    num_test_images = len(dataloader())

#%% initialize the reconstructor
reconstructor = IUNetReconstructor(operator=op, in_ch=1, img_size=None)

#%% load the desired model checkpoint

path_parts = ['..', 'experiments', dataset_name, 'iunet', server,
          model_version, 'checkpoints', chkp_name + '.ckpt']

chkp_path = os.path.join(*path_parts)
reconstructor.load_learned_params(path=chkp_path, checkpoint=True)

criterion = CINNNLLLoss(
    distribution=reconstructor.model.hparams.sample_distribution)

#%% move model to the GPU
reconstructor.model.to('cuda')
print("Number of Params: ", sum(p.numel() for p in 
      reconstructor.model.parameters() if p.requires_grad))

#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
reconstructor.samples_per_reco = samples_per_reco
reconstructor.max_samples_per_run = 100
pl.seed_everything(eval_seed)
reconstructor.model.eval()

# save report of the evaluation result
save_report = True

# check for the invertibility of the samples
check_inv = True
check_inv_gt = True

# calculate the NLL loss
check_loss = True
check_loss_gt = True

# plot the first three reconstructions & gt
plot_examples = True
plot_cond = True
plot_examples_gt = False

#%% evaluate the model 
if save_report:
    report_path_parts = path_parts[:-2]
    report_path_parts.append('benchmark')
    report_name = model_version + '_' + chkp_name + '_seed=' + str(eval_seed) + \
                    '_images=' + str(num_test_images) + \
                    '_samples=' + str(reconstructor.samples_per_reco)
    report_path_parts.append(report_name)
    report_path = os.path.join(*report_path_parts)
    Path(report_path).mkdir(parents=True, exist_ok=True)

recos = []
recos_cond = []
recos_std = []
losses = []
psnrs = []
ssims = []
psnrs_cond = []
ssims_cond = []
inv_psnr = []

with torch.no_grad():
    for i, batch in tqdm(zip(range(num_test_images),dataloader()), 
                         total=num_test_images):
        obs = batch[0]
        gt = batch[1]
        obs = obs.to('cuda')
        gt = gt.to('cuda')
        
        # create reconstruction from observation
        reco, reco_std = reconstructor._reconstruct(obs, return_std=True)
        if data_range is not None:
            reco = np.clip(reco, data_range[0], data_range[1])
        recos.append(reco)
        recos_std.append(reco_std)
        
        reco_cond = reconstructor.model.cond_net(obs)[0].cpu().numpy()
        reco_cond = reco_cond[0, 0, :gt.shape[-2], :gt.shape[-1]]
        recos_cond.append(reco_cond)

        # calculate quality metrics
        psnrs.append(PSNR(reco, gt.cpu().numpy()[0][0]))
        ssims.append(SSIM(reco, gt.cpu().numpy()[0][0]))
        
        psnrs_cond.append(PSNR(reco_cond, gt.cpu().numpy()[0][0]))
        ssims_cond.append(SSIM(reco_cond, gt.cpu().numpy()[0][0]))
        
        if check_inv or check_loss:
            # create torch tensor from reconstruction and observation
            reco_torch = torch.from_numpy(reco.asarray()[None, None].astype('float32')).to(
                    reconstructor.model.device)
            if data_range is not None:
                reco_torch = torch.clip(reco_torch, data_range[0], data_range[1])
            
            # calculate sample and log det Jacobian of the reconstruction 
            z, log_jac = reconstructor.model(cinn_input=reco_torch, 
                                             cond_input=obs,
                                             rev=False)

        if check_loss:
            # calculate the NLL loss of the sample
            loss = criterion(zz=z, log_jac=log_jac)
            losses.append(loss.detach().cpu().numpy())
        
        if check_inv:
            # calculate back from the sample to the reconstruction to test 
            # invertibility
            reco_inv = reconstructor.model(cinn_input=z, 
                                             cond_input=obs,
                                             rev=True)
            inv_psnr.append(PSNR(reco_inv.detach().cpu().numpy(), reco))

mean_psnr = np.mean(psnrs)
std_psnr = np.std(psnrs)
mean_psnr_cond = np.mean(psnrs_cond)
std_psnr_cond = np.std(psnrs_cond)
mean_ssim = np.mean(ssims)
std_ssim = np.std(ssims)
mean_ssim_cond = np.mean(ssims_cond)
std_ssim_cond = np.std(ssims_cond)
mean_loss = np.mean(losses) if check_loss else float("inf")
std_loss = np.std(losses) if check_loss else 0.
mean_inv_psnr = np.mean(inv_psnr) if check_inv else 0.
std_inv_psnr = np.std(inv_psnr) if check_inv else 0.

print('---')
print('Results network:')
print('mean psnr: {:f}'.format(mean_psnr))
print('std psnr: {:f}'.format(std_psnr))
print('mean ssim: {:f}'.format(mean_ssim))
print('std ssim: {:f}'.format(std_ssim))
if check_loss:
    print('mean loss: {:f}'.format(mean_loss))
    print('std loss: {:f}'.format(std_loss))
if check_inv:
    print('mean inversion psnr: {:f}'.format(mean_inv_psnr))
    print('std inversion psnr: {:f}'.format(std_inv_psnr))
print('---')
print('Results conditioning:')
print('mean psnr cond: {:f}'.format(mean_psnr_cond))
print('std psnr cond: {:f}'.format(std_psnr_cond))
print('mean ssim cond: {:f}'.format(mean_ssim_cond))
print('std ssim cond: {:f}'.format(std_ssim_cond))

#%% plot results for the first 3 images + reconstruction
if plot_examples:
    for i, batch in tqdm(zip(range(3), dataloader()),
                         total=3):
        gt = batch[1]
        gt = gt.numpy()[0][0]
        
        _, ax = plot_images([recos[i], gt, recos_std[i]],
                            fig_size=(10, 4), vrange='individual')
        ax[0].set_xlabel('PSNR: {:.2f}, SSIM: {:.2f}'.format(psnrs[i],
                                                             ssims[i]))
        ax[0].set_title('IUNetReconstructor')
        ax[1].set_title('ground truth')
        ax[2].set_title('IUNetReconstructor std')
        ax[0].figure.suptitle('test sample {:d}'.format(i))
        
        if save_report:
            img_save_path = os.path.join(report_path,
                                    'test sample {:d}'.format(i)+'.png')
                
            savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None, transparent=False,
                    bbox_inches=None, pad_inches=0.1, metadata=None)
            
#%% plot conditioning for the first image
if plot_cond:
    for i, batch in tqdm(zip(range(1), dataloader()),
                         total=1):
        y = batch[0]
        
        conds = reconstructor.model.cond_net(y.to('cuda'))
        
        conditions = list()
        for cond in conds:
            conditions.append(cond[0].detach().cpu().numpy())
        
        for k in range(len(conditions)):
            fig = plt.figure(figsize=(16., 16.))
            columns = min(8, conditions[k].shape[0])
            rows = int((7 + conditions[k].shape[0])/8)
            
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(rows, columns),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

            for ax, im in zip(grid, list(conditions[k])):
                # Iterating over the grid returns the Axes.
                ax.imshow(im)

        
            if save_report:
                img_save_path = os.path.join(report_path,
                                        'cond_' + str(k) + ' test sample {:d}'.format(i)+'.png')
                    
                savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=None, transparent=False,
                        bbox_inches=None, pad_inches=0.1, metadata=None)
    
#%% sanity check for invertibility and loss on 3 ground truth images
if check_inv_gt or check_loss_gt:
    recos_gt = []
    z_gt = []
    losses_gt = []
    inv_psnr_gt = []
    with torch.no_grad():
        for i, batch in tqdm(zip(range(3), dataloader()),
                             total=3):
            obs = batch[0]
            gt = batch[1]
            obs = obs.to('cuda')
            gt = gt.to('cuda')

            # reconstruct with gt to check invertibility
            z, log_jac = reconstructor.model(cinn_input=gt, 
                                             cond_input=obs,
                                             rev=False)
            z_gt.append(z.detach().cpu().numpy())
            
            if check_loss_gt:
                # calculate the NLL loss of the sample
                loss_gt = criterion(zz=z, log_jac=log_jac)
                losses_gt.append(loss_gt.detach().cpu().numpy())
            
            if check_inv_gt:
                 # calculate back from the sample to the reconstruction to test 
                 # invertibility
                reco_gt = reconstructor.model(cinn_input=z,
                                              cond_input=obs,
                                              rev=True,
                                              cut_ouput=True)[0][0]
                recos_gt.append(reco_gt.detach().cpu().numpy())
                inv_psnr_gt.append(PSNR(reco_gt.detach().cpu().numpy(),
                                        gt.detach().cpu().numpy()[0][0]))
        
    mean_loss_gt = np.mean(losses_gt) if check_loss_gt else float("inf")
    std_loss_gt = np.std(losses_gt) if check_loss_gt else 0.
    mean_inv_psnr_gt = np.mean(inv_psnr_gt) if check_inv_gt else 0.
    std_inv_psnr_gt = np.std(inv_psnr_gt) if check_inv_gt else 0.    
        
    print('---')
    print('Results on Ground Truth:')
    if check_loss_gt:
        print('mean loss gt: {:f}'.format(mean_loss_gt))
        print('std loss gt: {:f}'.format(std_loss_gt))
    if check_inv_gt:
        print('mean inversion psnr gt: {:f}'.format(mean_inv_psnr_gt))
        print('std inversion psnr gt: {:f}'.format(std_inv_psnr_gt))    
        
    if plot_examples_gt:
        for i, batch in tqdm(zip(range(3), dataloader()),
                             total=3):
            gt = batch[1]
            gt = gt.numpy()[0][0]
            
            _, ax = plot_images([recos_gt[i], gt],
                                fig_size=(10, 4), vrange='individual')
            ax[0].set_xlabel('PSNR: {:.2f}'.format(inv_psnr_gt[i]))
            ax[0].set_title('Reconstructed gt')
            ax[1].set_title('ground truth')
            ax[0].figure.suptitle('test sample {:d}'.format(i))
            
            if save_report:
                img_save_path = os.path.join(report_path,
                                    'test sample {:d}'.format(i)+'_gt.png')
                
                savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=None, transparent=False,
                        bbox_inches=None, pad_inches=0.1, metadata=None)

#%% create report file
if save_report:
    report_dict = {'settings': {'num_test_images': num_test_images,
                                'seed': eval_seed,
                                'samples_per_reco': 
                                    reconstructor.samples_per_reco},
                   'results': {'mean_psnr': float(mean_psnr),
                               'std_psnr': float(std_psnr),
                               'mean_ssim': float(mean_ssim),
                               'std_ssim': float(std_ssim),
                               'mean_loss': float(mean_loss),
                               'std_loss': float(std_loss),
                               'mean_inv_psnr': float(mean_inv_psnr),
                               'std_inv_psnr': float(std_inv_psnr)},
                   'results_cond': {'mean_psnr_cond': float(mean_psnr_cond),
                                    'std_psnr_cond': float(std_psnr_cond),
                                    'mean_ssim_cond': float(mean_ssim_cond),
                                    'std_ssim_cond': float(std_ssim_cond),
                       },
                   'results_gt': {'mean_loss_gt': float(mean_loss_gt),
                                  'std_loss_gt': float(std_loss_gt),
                                  'mean_inv_psnr_gt': float(mean_inv_psnr_gt),
                                  'std_inv_psnr_gt': float(std_inv_psnr_gt)}
        }
    
    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)
