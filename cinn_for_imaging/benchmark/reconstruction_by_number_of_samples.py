"""
Benchmark CINNReconstructor on the 'lodopab' test set.
"""

import os

from dival import reconstructors
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from dival.measure import PSNR, SSIM
from dival.util.plot import plot_images
from matplotlib.pyplot import savefig
from collections import defaultdict
import matplotlib.pyplot as plt 
import json

from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.reconstructors.iunet_reconstructor import IUNetReconstructor
from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule
#from cinn_for_imaging.datasets.lodopab_lpd.data_util import LoDoPaBLPDDataModule

model_type = "multi_scale" # iunet 


dataset = LoDoPaBDataModule(batch_size=1, impl='astra_cuda', sorted_by_patient=False)
dataset.prepare_data()
dataset.setup(stage='test')
ray_trafo = dataset.ray_trafo
num_test_images = 100 #len(dataset.test_dataloader())

#%% initialize the reconstructor
if model_type == "multi_scale":
    reconstructor = CINNReconstructor(
        ray_trafo=ray_trafo, 
        in_ch=1, 
        img_size=None)
else: 
    reconstructor = IUNetReconstructor(operator=ray_trafo, in_ch=1, img_size=None)

#%% load the desired model checkpoint
version = 'version_7' 
chkp_name = 'epoch=47-step=30671'
path_parts = ['..', 'experiments', 'lodopab', model_type, 'default', version, 'checkpoints', chkp_name + '.ckpt']
chkp_path = os.path.join(*path_parts)
reconstructor.load_learned_params(path=chkp_path, checkpoint=True)


criterion = CINNNLLLoss(
    distribution=reconstructor.model.hparams.sample_distribution)

#%% move model to the GPU
reconstructor.model.to('cuda:4')

print("Number of Params: ", sum(p.numel() for p in reconstructor.model.parameters() if p.requires_grad))
print("Sample distribution: ", reconstructor.sample_distribution)
#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
pl.seed_everything(eval_seed)
reconstructor.model.eval()

# save report of the evaluation result
save_report = False

# plot the first three reconstructions & gt
plot_examples = True
plot_examples_gt = False

#%% evaluate the model 
if save_report:
    report_path_parts = path_parts[:-2]
    report_path_parts.append('benchmark')
    report_name = version + '_' + chkp_name + '_seed=' + str(eval_seed) + \
                    '_images=' + str(num_test_images) + \
                    '_samples=' + str(reconstructor.samples_per_reco)
    report_path_parts.append(report_name)
    report_path = os.path.join(*report_path_parts)
    Path(report_path).mkdir(parents=True, exist_ok=True)

recos = []
recos_std = []
losses = []
psnrs = []
ssims = []
inv_psnr = []

num_samples = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]#[1] + np.arange(1, 11)*10
max_samples = np.max(num_samples)
print(num_samples, max_samples)
psnrs = defaultdict(list)
ssims = defaultdict(list)

with torch.no_grad():
    for i, batch in tqdm(zip(range(num_test_images),dataset.test_dataloader()), 
                         total=num_test_images):
        obs, gt = batch
        obs = obs.to('cuda:4')

        cond_input = reconstructor.model.cond_net(obs)

        # Draw random samples from the random distribution
        if reconstructor.sample_distribution == 'normal':
            z = torch.randn((max_samples, reconstructor.img_size[0]*reconstructor.img_size[1]),
                            device=reconstructor.model.device)
        # Draw random samples from the radial distribution
        elif reconstructor.sample_distribution == 'radial':                    
            z = torch.randn((max_samples, reconstructor.img_size[0]*reconstructor.img_size[1]),
                            device=reconstructor.model.device)
            z_norm = torch.norm(z,dim=1)
            r = torch.abs(torch.randn((max_samples,1), device=reconstructor.model.device))
            z = z/z_norm.view(-1,1)*r

        # use the cond_input calculated before the loop
        cond_input_rep = [torch.repeat_interleave(c, max_samples, dim=0) for c in cond_input]

        xgen, _ = reconstructor.model.cinn(z, c=cond_input_rep, rev=True)
        xgen = xgen.cpu()
        del z, cond_input_rep

        # cut output
        xgen = xgen[:,:,:reconstructor.op.domain.shape[0],:reconstructor.op.domain.shape[1]] 

        for n_samples in num_samples: 
            xmean = torch.mean(xgen[:n_samples, :, :, :], dim=0)
            reco = np.squeeze(xmean.detach().cpu().numpy())
            psnrs[n_samples].append(PSNR(reco, gt.cpu().numpy()[0][0]))
            ssims[n_samples].append(SSIM(reco, gt.cpu().numpy()[0][0]))

with open(model_type + "_" + reconstructor.sample_distribution + "_" + "num_images=" + str(num_test_images) + '_psnrs.json', 'w') as fp:
    json.dump(psnrs, fp)

with open(model_type + "_" + reconstructor.sample_distribution + "_" + "num_images=" + str(num_test_images) + '_ssims.json', 'w') as fp:
    json.dump(ssims, fp)


psnr_by_sample = [np.mean(psnrs[key]) for key in psnrs.keys()]
ssim_by_sample = [np.mean(ssims[key]) for key in ssims.keys()]

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(num_samples, psnr_by_sample)

ax2.plot(num_samples, ssim_by_sample)

plt.show()

"""
mean_psnr = np.mean(psnrs)
std_psnr = np.std(psnrs)
mean_ssim = np.mean(ssims)
std_ssim = np.std(ssims)


print('---')
print('Results:')
print('mean psnr: {:f}'.format(mean_psnr))
print('std psnr: {:f}'.format(std_psnr))
print('mean ssim: {:f}'.format(mean_ssim))
print('std ssim: {:f}'.format(std_ssim))


#%% create report file
if save_report:
    report_dict = {'settings': {'num_test_images': num_test_images,
                                'seed': eval_seed,
                                'samples_per_reco': 
                                    reconstructor.samples_per_reco},
                   'results': {'mean_psnr': float(mean_psnr),
                               'std_psnr': float(std_psnr),
                               'mean_ssim': float(mean_ssim),
                               'std_ssim': float(std_ssim)
                               },
 
        }
    
    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)
"""