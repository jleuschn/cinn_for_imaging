"""
Train CINNReconstructor on 'lodopab'.
"""

import os

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.datasets.lodopab.data_util import LoDoPaBDataModule



#%% path for logging and saving the model
experiment_name = 'multi_scale'
path_parts = ['..', 'experiments', 'lodopab', experiment_name]
log_dir = os.path.join(*path_parts)

#%% setup the LoDoPaB dataset
dataset = LoDoPaBDataModule(impl='astra_cuda', sorted_by_patient=True,
                            num_data_loader_workers=8)
dataset.prepare_data()
dataset.setup()
ray_trafo = dataset.ray_trafo

#%% configure the Pytorch Lightning trainer. 
# Visit https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
# for all available trainer options.

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
                'default_root_dir': log_dir,
                'callbacks': [lr_monitor, checkpoint_callback],
                'benchmark': False,
                'fast_dev_run': False,
                'limit_train_batches': 0.25,
                #'limit_val_batches': 0.1,
                'gradient_clip_val': 1.0,
                #'resume_from_checkpoint': os.path.join(log_dir, 'default', 'version_1', 'checkpoints', 'epoch=18-step=48620.ckpt'),
                'logger': tb_logger}

#%% create the reconstructor
reconstructor = CINNReconstructor(
    ray_trafo=ray_trafo, 
    in_ch=1, 
    img_size=None,
    max_samples_per_run=100,
    trainer_args=trainer_args,
    downsample_levels=5,
    log_dir=log_dir)

#%% change some of the hyperparameters of the reconstructor
reconstructor.batch_size = 14
reconstructor.epochs = 500

reconstructor.conditioning = 'ResNetCond'
reconstructor.coupling = 'affine'
reconstructor.cond_fc_size = 64
reconstructor.num_blocks = 5
reconstructor.num_fc = 2
reconstructor.permutation = '1x1'
reconstructor.cond_conv_channels = [4, 16, 32, 64, 64, 32]
reconstructor.train_noise = (0., 0.005)
reconstructor.sample_distribution = 'normal'


#%% train the reconstructor. Checkpointing and logging is enabled by default.
reconstructor.train(dataset)

#version = 'version_0' 
#chkp_name = 'epoch=17-step=46061'


#path_parts = ['..', 'experiments', 'lodopab', experiment_name, 'default',
#             version, 'checkpoints', chkp_name + '.ckpt']

#chkp_path = os.path.join(*path_parts)
#reconstructor.train(dataset,checkpoint_path=chkp_path)

