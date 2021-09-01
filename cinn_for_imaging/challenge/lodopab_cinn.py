"""
Evaluate CINNReconstructor on the 'lodopab' challenge set.
"""

import os

from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import yaml
import pytorch_lightning as pl
from dival import get_standard_dataset
from dival.util.plot import plot_images
from matplotlib.pyplot import savefig

from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.datasets.lodopab_challenge.challenge_set import (
        config, generator, NUM_IMAGES)
from cinn_for_imaging.datasets.lodopab_challenge.submission import (
        save_reconstruction, pack_submission)


#%% configure the dataset
config['data_path'] = '/localdata/lodopab_challenge_set'  # TODO adapt
IMPL = 'astra_cuda'
dataset = get_standard_dataset('lodopab', impl=IMPL)
ray_trafo = dataset.get_ray_trafo(impl=IMPL)

#%% initialize the reconstructor
reconstructor = CINNReconstructor(
    ray_trafo=ray_trafo, 
    in_ch=1, 
    img_size=None)

#%% load the desired model checkpoint
experiment_name = 'lodopab'
version = 'version_0' 
chkp_name = 'epoch=36'
path_parts = ['..', 'experiments', 'lodopab', 'default',
              version, 'checkpoints', chkp_name + '.ckpt']
chkp_path = os.path.join(*path_parts)
reconstructor.load_learned_params(path=chkp_path, checkpoint=True)

criterion = CINNNLLLoss(
    distribution=reconstructor.model.hparams.sample_distribution)


print("Number of Params: ", sum(p.numel() for p in reconstructor.model.parameters() if p.requires_grad))

#%% move model to the GPU
reconstructor.model.to('cuda')

#%% settings for the evaluation - deactivate seed if needed
eval_seed = 42
reconstructor.samples_per_reco = 100
reconstructor.max_samples_per_run = 100
pl.seed_everything(eval_seed)
reconstructor.model.eval()

# save report of the evaluation result
save_report = True

# calculate the NLL loss
check_loss = True

# plot the first three reconstructions
plot_examples = True

#%% evaluate the model 
if save_report:
    report_path_parts = path_parts[:-2]
    report_path_parts.append('challenge')
    report_name = version + '_' + chkp_name + '_seed=' + str(eval_seed) + \
                    '_images='  + str(NUM_IMAGES) + \
                    '_samples=' + str(reconstructor.samples_per_reco)
    report_path_parts.append(report_name)
    report_path = os.path.join(*report_path_parts)
    Path(report_path).mkdir(parents=True, exist_ok=True)

output_path_parts = path_parts[:-2]
output_path_parts.append('challenge')
output_name = version + '_' + chkp_name + '_seed=' + str(eval_seed) + \
                    '_images='  + str(NUM_IMAGES) + \
                    '_samples=' + str(reconstructor.samples_per_reco)
output_path_parts.append(report_name)
output_path = os.path.join(*report_path_parts)
Path(output_path).mkdir(parents=True, exist_ok=True)


recos = []
losses = []
with torch.no_grad():
    for i, obs in enumerate(tqdm(generator(), total=NUM_IMAGES)):
        # create reconstruction from observation and save them
        reco = reconstructor.reconstruct(obs)
        save_reconstruction(output_path, i, reco)
        recos.append(reco)
        
        if check_loss:
            # create torch tensor from reconstruction and observation
            reco_torch = torch.from_numpy(reco.asarray()[None, None]).to(
                    reconstructor.model.device)
            obs_torch = torch.from_numpy(np.asarray(obs)[None, None]).to(
                            reconstructor.model.device)
            
            # calculate sample and log det Jacobian of the reconstruction 
            z, log_jac = reconstructor.model(cinn_input=reco_torch, 
                                             cond_input=obs_torch,
                                             rev=False)

            # calculate the NLL loss of the sample
            loss = criterion(zz=z, log_jac=log_jac)
            losses.append(loss.detach().cpu().numpy())

pack_submission(output_path)
        
mean_loss = np.mean(losses) if check_loss else None
std_loss = np.std(losses) if check_loss else None

print('---')
print('Results:')
if check_loss:
    print('mean loss: {:f}'.format(mean_loss))
    print('std loss: {:f}'.format(std_loss))

#%% plot results for the first 3 images + reconstruction
if plot_examples:
    for i in range(3):
        _, ax = plot_images([recos[i]],
                            fig_size=(10, 4), vrange='individual')
        ax[0].set_title('CINNReconstructor')
        ax[0].figure.suptitle('challenge sample {:d}'.format(i))
        
        if save_report:
            img_save_path = os.path.join(report_path,
                                    'challenge sample {:d}'.format(i)+'.png')
                
            savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None, transparent=False,
                    bbox_inches=None, pad_inches=0.1, metadata=None)

#%% create report file
if save_report:
    report_dict = {'settings': {'num_challenge_images': NUM_IMAGES,
                                'seed': eval_seed,
                                'samples_per_reco': 
                                    reconstructor.samples_per_reco},
                   'results': {'mean_loss': float(mean_loss),
                               'std_loss': float(std_loss)}
        }
    
    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)
