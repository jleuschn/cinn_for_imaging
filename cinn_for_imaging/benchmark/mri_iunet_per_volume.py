"""
Validation IUNetReconstructor on the 'fastmri' data set.
"""

import os

from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import yaml
import matplotlib.pyplot as plt 

from cinn_for_imaging.reconstructors.iunet_reconstructor import IUNetReconstructor
from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule
from cinn_for_imaging.datasets.fast_mri.submission import save_reconstructions

from skimage.metrics import peak_signal_noise_ratio, structural_similarity



def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() 
    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


save_report = True

#%% configure the dataset
dataset = FastMRIDataModule(batch_size=1, num_data_loader_workers=1)
dataset.prepare_data()
dataset.setup()
num_val_images = len(dataset.val_dataloader())

op = dataset.operator

#%% initialize the reconstructor
reconstructor = IUNetReconstructor(
    operator=op,
    in_ch=1, 
    img_size=None,
    max_samples_per_run=100)
#%% load the desired model checkpoint
# version = 'version_9'  # moriarty
# chkp_name = 'epoch=21-step=63711'
# version = 'version_10'  # moriarty
# chkp_name = 'epoch=26-step=78191'
# version = 'version_14'  # moriarty
# chkp_name = 'epoch=27-step=81087'
version = 'version_15'  # moriarty
chkp_name = 'epoch=29-step=86879'
path_parts = ['..', 'experiments', 'fast_mri', 'iunet', 'default', version, 'checkpoints', chkp_name + '.ckpt']



chkp_path = os.path.join(*path_parts)
reconstructor.load_learned_params(path=chkp_path, checkpoint=True)
#reconstructor.init_model()
#reconstructor.model.init_params()
#%% move model to the GPU
reconstructor.model.to('cuda')

print("Number of Params:")
print("cINN: ", sum(p.numel() for p in reconstructor.model.cinn.parameters() if p.requires_grad))
print("CondNet: ", sum(p.numel() for p in reconstructor.model.cond_net.parameters() if p.requires_grad))
print("Full number: ", sum(p.numel() for p in reconstructor.model.parameters() if p.requires_grad))
print("\n")

#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
reconstructor.samples_per_reco = 100
reconstructor.max_samples_per_run = 100
pl.seed_everything(eval_seed)
reconstructor.model.eval()


if save_report:
    report_path_parts = path_parts[:-2]
    report_path_parts.append('benchmark_per_volume')
    report_name = version + '_' + chkp_name + '_seed=' + str(eval_seed) + \
                    '_images=' + str(num_val_images) + \
                    '_samples=' + str(reconstructor.samples_per_reco)
    report_path_parts.append(report_name)
    report_path = os.path.join(*report_path_parts)
    Path(report_path).mkdir(parents=True, exist_ok=True)


reconstructions_pd = defaultdict(list)
targets_pd = defaultdict(list)

reconstructions_pdfs = defaultdict(list)
targets_pdfs = defaultdict(list)
with torch.no_grad():
    for i, batch in tqdm(zip(range(num_val_images),dataset.val_dataloader()), 
                         total=num_val_images):
        obs, target, mean, std, fname, slice_num, acquisition = batch
        obs = obs.to('cuda')
        #print(fname)
        #print(filename, slice_num)
        
        # create reconstruction from observation
        reco, reco_std = reconstructor._reconstruct(obs,return_std=True)
        reco = np.asarray(reco)
        target = np.asarray(target[0][0])
        
        reco = reco * std.numpy() + mean.numpy() 
        target = target * std.numpy() + mean.numpy() 
        
        reco = np.clip(reco, a_min=0, a_max=0.005) # max value in train data: 0.0024660237
        """
        fig, (ax1, ax2) = plt.subplots(1,2)

        im = ax1.imshow(reco, cmap="gray")
        fig.colorbar(im, ax=ax1)
        #axes[0,1].hist(image[0,0,:,:].ravel(), bins="auto")

        im = ax2.imshow(target, cmap="gray")
        fig.colorbar(im, ax=ax2)

        #axes[1,1].hist(gt[0,0,:,:].ravel(), bins="auto")
        plt.show()
        """
        if acquisition[0] == 'CORPDFS_FBK':
            reconstructions_pdfs[fname[0]].append((int(slice_num[0]), reco))
            targets_pdfs[fname[0]].append((int(slice_num[0]), target))
        else: 
            reconstructions_pd[fname[0]].append((int(slice_num[0]), reco))
            targets_pd[fname[0]].append((int(slice_num[0]), target))




# save outputs
for fname in reconstructions_pd:
    reconstructions_pd[fname] = np.stack([out for _, out in sorted(reconstructions_pd[fname])])
    targets_pd[fname] = np.stack([out for _, out in sorted(targets_pd[fname])])

# save outputs
for fname in reconstructions_pdfs:
    reconstructions_pdfs[fname] = np.stack([out for _, out in sorted(reconstructions_pdfs[fname])])
    targets_pdfs[fname] = np.stack([out for _, out in sorted(targets_pdfs[fname])])



psnrs_pd = [] 
ssims_pd = []
for fname in reconstructions_pd:
    psnrs_pd.append(psnr(targets_pd[fname], reconstructions_pd[fname]))
    ssims_pd.append(ssim(targets_pd[fname], reconstructions_pd[fname]))

psnrs_pdfs = [] 
ssims_pdfs = []
for fname in reconstructions_pdfs:
    psnrs_pdfs.append(psnr(targets_pdfs[fname], reconstructions_pdfs[fname]))
    ssims_pdfs.append(ssim(targets_pdfs[fname], reconstructions_pdfs[fname]))

mean_psnr_pd = np.mean(psnrs_pd)
std_psnr_pd = np.std(psnrs_pd)
mean_ssim_pd = np.mean(ssims_pd)
std_ssim_pd = np.std(ssims_pd)

mean_psnr_pdfs = np.mean(psnrs_pdfs)
std_psnr_pdfs = np.std(psnrs_pdfs)
mean_ssim_pdfs = np.mean(ssims_pdfs)
std_ssim_pdfs = np.std(ssims_pdfs)



print('---')
print('Results for PD:')
print('mean psnr: {:f}'.format(mean_psnr_pd))
print('std psnr: {:f}'.format(std_psnr_pd))
print('mean ssim: {:f}'.format(mean_ssim_pd))
print('std ssim: {:f}'.format(std_ssim_pd))

print('---')
print('Results for PDFS:')
print('mean psnr: {:f}'.format(mean_psnr_pdfs))
print('std psnr: {:f}'.format(std_psnr_pdfs))
print('mean ssim: {:f}'.format(mean_ssim_pdfs))
print('std ssim: {:f}'.format(std_ssim_pdfs))


#%% create report file
if save_report:
    report_dict = {'settings': {'num_val_images': num_val_images,
                                'seed': eval_seed,
                                'samples_per_reco': 
                                    reconstructor.samples_per_reco},
                   'results_PD': {'mean_psnr': float(mean_psnr_pd),
                               'std_psnr': float(std_psnr_pd),
                               'mean_ssim': float(mean_ssim_pd),
                               'std_ssim': float(std_ssim_pd)},
                   'results_PDFS': {'mean_psnr': float(mean_psnr_pdfs),
                               'std_psnr': float(std_psnr_pdfs),
                               'mean_ssim': float(mean_ssim_pdfs),
                               'std_ssim': float(std_ssim_pdfs)}
        }
    
    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)
