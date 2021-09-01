"""
Train CINNReconstructor on 'fastMRI'.
"""

import os

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from cinn_for_imaging.reconstructors.mri_reconstructor import CINNReconstructor
from cinn_for_imaging.datasets.fast_mri.data_util import FastMRIDataModule


#%% setup the dataset
dataset = FastMRIDataModule(num_data_loader_workers=os.cpu_count(),
                            batch_size=8)
dataset.prepare_data()
dataset.setup()

#%% path for logging and saving the model
experiment_name = 'cinn'
path_parts = ['..', 'experiments', 'fast_mri', experiment_name]
log_dir = os.path.join(*path_parts)

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

trainer_args = {'accelerator': 'ddp',
                'gpus': [0],
                'default_root_dir': log_dir,
                'callbacks': [lr_monitor, checkpoint_callback],
                'benchmark': True,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                'precision': 32,
                'limit_train_batches': 0.25,
                'limit_val_batches': 0.25,
                'terminate_on_nan': True}#,
                #'limit_train_batches': 0.1,
                #'limit_val_batches': 0.1}#,
                #'overfit_batches':10}

#%% create the reconstructor
reconstructor = CINNReconstructor(
    in_ch=1, 
    img_size=(320, 320),
    max_samples_per_run=100,
    trainer_args=trainer_args,
    log_dir=log_dir)

#%% change some of the hyperparameters of the reconstructor
reconstructor.batch_size = 12
reconstructor.epochs = 500

reconstructor.sample_distribution = 'normal'
reconstructor.conditioning = 'ResNetCond'
reconstructor.coupling = 'affine'
reconstructor.cond_fc_size = 64
reconstructor.num_blocks = 5
reconstructor.num_fc = 3  # TODO does nothing, replace by reconstructor.num_fc_blocks (= 2)?
reconstructor.permutation = '1x1'
reconstructor.cond_conv_channels = [4, 16, 16, 16, 32, 32, 32]
reconstructor.train_noise = (0., 0.005)
reconstructor.use_act_norm = True

#%% train the reconstructor. Checkpointing and logging is enabled by default.
reconstructor.train(dataset)
