
import os 

from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule

import matplotlib.pyplot as plt 
import matplotlib
import torch 
import torch.nn as nn 
from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule
from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule
from tqdm import tqdm


class FBPModule(torch.nn.Module):
    """
    Torch module of the filtered back-projection (FBP).

    Methods
    -------
    forward(x)
        Compute the FBP.
        
    """
    
    def __init__(self, ray_trafo, filter_type='Hann',
                 frequency_scaling=1.):
        super().__init__()
        self.ray_trafo = ray_trafo
        filter_op = fbp_filter_op(self.ray_trafo,
                          filter_type=filter_type,
                          frequency_scaling=frequency_scaling)
        self.filter_mod = OperatorModule(filter_op)
        self.ray_trafo_adjoint_mod = (
            TorchRayTrafoParallel2DAdjointModule(self.ray_trafo))
        
    def forward(self, x):
        x = self.filter_mod(x)
        x = self.ray_trafo_adjoint_mod(x)
        return x




#%% setup the dataset
dataset_mri = FastMRIDataModule(num_data_loader_workers=os.cpu_count(),
                            batch_size=1)
dataset_mri.prepare_data()
dataset_mri.setup()


#%% setup the LoDoPaB dataset
dataset_ct = LoDoPaBDataModule(impl='astra_cuda', sorted_by_patient=True,
                            num_data_loader_workers=8, batch_size=1)
dataset_ct.prepare_data()
dataset_ct.setup()
ray_trafo = dataset_ct.ray_trafo

fbp = FBPModule(ray_trafo)
ray_trafo_op = OperatorModule(ray_trafo)


num_test_images = 10

"""
for i, batch in tqdm(zip(range(num_test_images),dataset_mri.train_dataloader()), 
                        total=num_test_images):

    rec = batch[0]
    gt = batch[1]
    masked_kspace = batch[2]

    print(rec.shape, gt.shape, masked_kspace.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    current_cmap = plt.get_cmap('gray').copy()#matplotlib.cm.get_cmap().copy()
    current_cmap.set_bad(color='black')
    ax1.imshow(rec[0][0], cmap="gray")
    ax2.imshow(gt[0][0], cmap="gray")
    ax3.imshow(20*torch.log(torch.abs(torch.view_as_complex(masked_kspace)[0][0])), cmap=current_cmap,interpolation='none')
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.show()
"""


for i, batch in tqdm(zip(range(num_test_images),dataset_ct.train_dataloader()), 
                        total=num_test_images):

    y, x = batch 
    print(x.shape, y.shape)
    fbp_reco = fbp(y)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1.imshow(x[0][0], cmap="gray")
    ax2.imshow(y[0][0], cmap="gray")
    ax3.imshow(fbp_reco[0][0], cmap="gray")

    #ax3.imshow(20*torch.log(torch.abs(torch.view_as_complex(masked_kspace)[0][0])), cmap=current_cmap,interpolation='none')
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.show()
