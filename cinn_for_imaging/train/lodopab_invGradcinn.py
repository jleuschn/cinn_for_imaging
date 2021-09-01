"""
Train CINNReconstructor on 'lodopab'.
"""

import os
import odl 
import torch
from dival import get_standard_dataset, reconstructors
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from cinn_for_imaging.reconstructors.gradient_descent_reconstructor import CINNReconstructor
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule


#%% path for logging and saving the model
experiment_name = 'invGD'
path_parts = ['..', 'experiments', experiment_name]
log_dir = os.path.join(*path_parts)

#%% setup the LoDoPaB dataset
dataset = LoDoPaBDataModule(impl='astra_cuda',sorted_by_patient=True, num_data_loader_workers=0)
dataset.prepare_data()
dataset.setup()
ray_trafo = dataset.ray_trafo

step_size = 0.1#torch.tensor(1/(odl.power_method_opnorm(ray_trafo)**2*2)).float()

#%% configure the Pytorch Lightning trainer. 
# Visit https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
# for all available trainer options.

checkpoint_callback = ModelCheckpoint(
    filepath=None,
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

lr_monitor = LearningRateMonitor(logging_interval=None) 


tb_logger = pl_loggers.TensorBoardLogger(log_dir)

trainer_args = {'accelerator': 'ddp',
                'gpus': [0, 1],
                'default_root_dir': log_dir,
                'checkpoint_callback': checkpoint_callback,
                'callbacks': [lr_monitor],
                'benchmark': False,
                'fast_dev_run': False,
                'gradient_clip_val': 10.0,
                'logger': tb_logger,
                'limit_train_batches': 0.05,
                'limit_val_batches': 0.01,
                'terminate_on_nan': True}
                # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)

#%% create the reconstructor
reconstructor = CINNReconstructor(
    ray_trafo=ray_trafo, 
    in_ch=1, 
    img_size=(362, 362),
    step_size = step_size,
    max_samples_per_run=100,
    trainer_args=trainer_args,
    log_dir=log_dir)

#%% change some of the hyperparameters of the reconstructor
reconstructor.batch_size = 2
reconstructor.epochs = 500
reconstructor.downsampling = 'reshape'
reconstructor.upsampling = 'reshape'
reconstructor.actnorm = False 
reconstructor.permutation = "PermuteRandom"
reconstructor.unrolling_steps = 6
reconstructor.num_blocks = 3 
reconstructor.coupling = 'additive'

#%% train the reconstructor. Checkpointing and logging is enabled by default.
reconstructor.train(dataset)

