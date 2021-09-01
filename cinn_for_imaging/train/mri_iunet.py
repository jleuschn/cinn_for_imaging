"""
Train IUNetReconstructor on 'fastMRI'.
"""

import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from cinn_for_imaging.reconstructors.iunet_reconstructor import IUNetReconstructor
from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule

#%% setup the dataset
dataset = FastMRIDataModule(num_data_loader_workers=8)
dataset.prepare_data()
dataset.setup()

op = dataset.operator

#%% path for logging and saving the model
experiment_name = 'iunet'
path_parts = ['..', 'experiments', 'fast_mri', experiment_name]
log_dir = os.path.join(*path_parts)

#%% configure the Pytorch Lightning trainer. 
# Visit https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
# for all available trainer options.

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    verbose=True,
    monitor='val_loss',
    mode='min',
)

tb_logger = pl_loggers.TensorBoardLogger(log_dir)

trainer_args = {'accelerator': 'ddp',
                'gpus': [0],
                'default_root_dir': log_dir,
                'callbacks': [checkpoint_callback],
                'benchmark': True,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                'precision': 32,
                # 'limit_train_batches': 0.1,
                'terminate_on_nan': True}

#%% create the reconstructor
reconstructor = IUNetReconstructor(
    operator=op,
    in_ch=1,
    img_size=None,
    downsample_levels=5,
    conditioning="fft",
    max_samples_per_run=100,
    trainer_args=trainer_args,
    log_dir=log_dir)

#%% change some of the hyperparameters of the reconstructor
reconstructor.batch_size = 12
reconstructor.epochs = 500
reconstructor.downsampling = 'standard'
reconstructor.sample_distribution = 'normal'
reconstructor.torch_manual_seed = None
reconstructor.train_cond = 0.
reconstructor.weight_decay = 0.
reconstructor.coupling = 'additive'  # 'affine', 'additive', 'allInOne', 'RNVP'
reconstructor.train_noise = (0., 0.) # (0., 0.), (0., 0.06)
reconstructor.depth = 19
reconstructor.clamp_all = False
reconstructor.permute = True
reconstructor.permute_type = '1x1'
reconstructor.special_init = True
reconstructor.normalize_inn = True

#%% train the reconstructor. Checkpointing and logging is enabled by default.
reconstructor.train(dataset)
