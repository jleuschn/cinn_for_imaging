"""
Train IUNetReconstructor on 'lodopab'.
"""

import os

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
#from pytorch_lightning.plugins import DDPPlugin

from cinn_for_imaging.reconstructors.iunet_reconstructor import IUNetReconstructor
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule


#%% path for logging and saving the model
experiment_name = 'iunet'
path_parts = ['..', 'experiments', 'lodopab', experiment_name]
log_dir = os.path.join(*path_parts)

#%% setup the LoDoPaB dataset
dataset = LoDoPaBDataModule(impl='astra_cuda', sorted_by_patient=True,
                            num_data_loader_workers=8)
dataset.prepare_data()
dataset.setup()
ray_trafo = dataset.ray_trafo

#%% configure the Pytorch Lightning trainer.

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
)

lr_monitor = LearningRateMonitor(logging_interval=None) 

tb_logger = pl_loggers.TensorBoardLogger(log_dir)

trainer_args = {'distributed_backend': 'ddp',
                'gpus': [0],
                #'plugins' : DDPPlugin(find_unused_parameters=True),
                'default_root_dir': log_dir,
                'callbacks': [lr_monitor, checkpoint_callback],
                'benchmark': False,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                #'resume_from_checkpoint': os.path.join(log_dir, 'default', 'version_1', 'checkpoints', 'epoch=18-step=21260.ckpt'),
                'limit_train_batches': 0.25}

#%% create the reconstructor
reconstructor = IUNetReconstructor(
    operator=ray_trafo,
    in_ch=1,
    img_size=None,
    downsample_levels=5,
    conditioning="fbp",
    max_samples_per_run=100,
    trainer_args=trainer_args,
    log_dir=log_dir)

#%% change some of the hyperparameters of the reconstructor
reconstructor.batch_size = 6
reconstructor.epochs = 500
reconstructor.downsampling = 'standard'
reconstructor.sample_distribution = 'normal'
reconstructor.torch_manual_seed = None
reconstructor.train_cond = 1.
reconstructor.weight_decay = 0.
reconstructor.coupling = 'additive'  # 'affine', 'additive', 'allInOne', 'RNVP'
reconstructor.train_noise = (0., 0.005) # (0., 0.), (0., 0.005), (0, 0.01)
reconstructor.depth = 22
reconstructor.clamp_all = False
reconstructor.permute = True
reconstructor.special_init = True
reconstructor.normalize_inn = True
reconstructor.permute_type = '1x1' # '1x1'

#%% train the reconstructor. Checkpointing and logging is enabled by default.
reconstructor.train(dataset)

#version = 'version_1' 
#chkp_name = 'epoch=3'
