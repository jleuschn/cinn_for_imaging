# -*- coding: utf-8 -*-
from operator import truediv
from warnings import warn
from copy import deepcopy

import torch
import numpy as np
from dival.reconstructors import StandardLearnedReconstructor
import odl
import pytorch_lightning as pl

from cinn_for_imaging.reconstructors.networks.cinn_mri import CINN

class CINNReconstructor(StandardLearnedReconstructor):
    """
    Dival reconstructor class for the cINN network.
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'epochs': {
            'default': 500,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'weight_decay': {
            'default': 1e-7,
            'retrain': True
        },
        'batch_size': {
            'default': 10,
            'retrain': True
        },
        'downsample_levels': {
            'default': 6,
            'retrain': True, 
            'choices': [4,5,6]
        },
        'conditioning': {
            'default': 'ResNetCond',
            'retrain': True,
            'choices': ["ResNetCond", "SimpleCondNet", "AvgPoolCondNet"]
        },
        'downsampling': {
            'default': 'invertible',
            'retrain': True,
            'choices': ['invertible', 'haar', 'checkerboard ']
        },
        'sample_distribution': {
            'default': 'normal',
            'retrain': True,
            'choices' : ['normal', 'radial']
        },
        'samples_per_reco': {
            'default': 100,
            'retrain': False
        },
        'coupling' : {
            'default': 'affine',
            'retrain': True,
            'choices': ['affine', 'additive']
        }, 
        'use_fc_block' : {
            'default' :  True, 
            'retrain': True, 
            'choices': [False, True]
        }, 
        'use_act_norm': {
            'default':False,
            'retrain':True, 
            'choices': [False, True]
        }, 
        'permutation': {
            'default': "1x1",
            'retrain': True,
            'choices': ["1x1", "PermuteRandom"]
        }, 
        'cond_conv_channels': {
            'default': [4, 16, 32, 64, 64, 32],
            'retrain': True
        }, 
        'cond_fc_size': {
            'default': 64,
            'retrain':True
        }, 
        'num_blocks': {
            'default': 6,
            'retrain': True
        },
        'num_fc_blocks': {
            'default': 2,
            'retrain': True
        }, 
        'train_noise': {
            'default': [0., 0.], 
            'retrain': True
        }
    })

    def __init__(self, in_ch: int = 1, img_size=(320, 320),
                 max_samples_per_run: int = 100,
                 trainer_args:dict = {'distributed_backend': 'ddp',
                                      'gpus': [0]},
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : TYPE
            Ray transformation (forward operator).
        in_ch : int, optional
            Number of input channels.
            The default is 1.
        img_size : tuple of int, optional
            Internal size of the reconstructed image. This must be divisible by
            2**i for i=1,...,downsample_levels. By choosing None an optimal
            image size will be determined automatically.
            The default is None.
        max_samples_per_run : int, optional
            Max number of samples for a single run of the network. Adapt to the
            memory of your GPU. To reach the desired samples_per_reco,
            multiple runs of the network will automatically be performed.
            The default is 100.
        trainer_args : dict, optional
            Arguments for the Pytorch Trainer.
            The defaults are distributed_backend='ddp' and gpus=[0]
        Returns
        -------
        None.

        """
        
        super().__init__(op=odl.operator.default_ops.IdentityOperator(odl.uniform_discr([0, 0], [1, 1], shape=(320, 320))), **kwargs)
        

        self.in_ch = in_ch
        self.max_samples_per_run = max_samples_per_run

        self.trainer_args = trainer_args
        
        self.trainer = pl.Trainer(max_epochs=self.epochs, **self.trainer_args)
        
        self.img_size = img_size
            
                
    def init_model(self):
        """
        Initialize the model.

        Returns
        -------
        None.

        """
        self.model = CINN(in_ch=self.in_ch,
                          img_size=self.img_size,
                          optimizer_args = {
                              'lr': self.lr,
                              'weight_decay': self.weight_decay},
                          conditioning=self.conditioning,
                          downsample_levels=self.downsample_levels,
                          downsampling=self.downsampling, 
                          sample_distribution=self.sample_distribution,
                          coupling=self.coupling,
                          use_fc_block=self.use_fc_block,
                          use_act_norm=self.use_act_norm, 
                          permutation=self.permutation,
                          cond_conv_channels=self.cond_conv_channels,
                          cond_fc_size=self.cond_fc_size, 
                          num_blocks=self.num_blocks,
                          num_fc_blocks=self.num_fc_blocks,
                          train_noise=self.train_noise)

    def train(self, dataset, checkpoint_path:str = None):
        """
        The training logic uses Pytorch Lightning.

        Parameters
        ----------
        dataset : LightningDataModule
            Pytorch Lighnting data module with (measurements, gt).
        checkpoint_path : str, optional
            Path to a .ckpt file to continue training. Will be ignored if None.
            The default is None.

        Returns
        -------
        None.

        """
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        if self.torch_manual_seed:
            pl.seed_everything(self.torch_manual_seed)
        
        # initialize model before training from checkpoint or from scratch
        if checkpoint_path:
            self.load_learned_params(path=checkpoint_path, checkpoint=True,
                                     strict=False)
        else:
            self.init_model()
            
        # set the batch size
        dataset.batch_size = self.batch_size
        
        # train pytorch lightning model
        self.trainer.fit(self.model, datamodule=dataset)

    def _reconstruct(self, observation, return_torch: bool = False,
                     return_std: bool = False, cut_ouput: bool = True,
                     *args, **kwargs):
        """
        Create a reconstruction from the observation with the cINN
        Reconstructor. 
        
        The batch size must be 1!

        Parameters
        ----------
        observation : numpy array or torch tensor
            Observation data (measurement).
        return_torch : bool, optional
            The method will return an ODL element if False with just the
            image size. If True a torch tensor will be returned. In this case,
            singleton dimensions are NOT removed!
            The default is False.
        return_std : bool, optional
            Also return the standard deviation between the samples used for 
            the reconstruction. This will slow down the reconstruction! Since
            all intermediate results are required, the std is computed on
            the cpu to reduce GPU memory consumption.
            The default is False.
        cut_ouput : bool, optional
            Cut the output to the original size of the ground truth data.
            The default is True.

        Returns
        -------
        xmean : ODL element or torch tensor
            Reconstruction based on the mean of the samples.
        xstd : ODL element or torch tensor, optional
            Standard deviation between the individual samples. Only returned
            if return_std == True

        """
        
        # initialize xmean
        xmean = 0
        
        # create torch tensor if necessary and put in on the same device as
        # the cINN model
        if not torch.is_tensor(observation):
                observation = torch.from_numpy(
                    np.asarray(observation)[None, None]).to(self.model.device)
        
        if observation.shape[0] > 1:
            warn('Batch size greater than 1 is not supported in the' + 
                 'reconstruction process!')
        
        # run the reconstruction process over multiple samples and calculate 
        # mean and standard deviation
        with torch.no_grad():
            xgen_list = []
            
            # Limit the number of max samples for a single run of the network
            # to self.max_samples_per_run
            samples_per_run =  np.arange(0, self.samples_per_reco,
                                         self.max_samples_per_run)
            samples_per_run = list(np.append(
                samples_per_run, self.samples_per_reco)[1:] - samples_per_run)
            
            # calculate condNet
            cond_input = self.model.cond_net(observation)

            for num_samples in samples_per_run:
                # Draw random samples from the random distribution
                if self.sample_distribution == 'normal':
                    z = torch.randn((num_samples,
                                     self.img_size[0]*self.img_size[1]),
                                    device=self.model.device)
                # Draw random samples from the radial distribution
                elif self.sample_distribution == 'radial':                    
                    z = torch.randn((num_samples,
                                     self.img_size[0]*self.img_size[1]),
                                    device=self.model.device)
                    z_norm = torch.norm(z,dim=1)
                    r = torch.abs(torch.randn((num_samples,1), device=self.model.device))
                    z = z/z_norm.view(-1,1)*r
                else:
                    raise ValueError("Sample distribution must be', "
                         "'normal' or 'radial', not '{}'".format(self.sample_distribution))
                    
                #obs_rep = torch.repeat_interleave(observation,
                #                                  num_samples,
                #                                  dim = 0)

                # TODO: There is some strange memory problem with the reverse
                # direction. Might be the self.model.cinn._buffers of the cINN 
                # Reversible GraphNet. The memory is only free after one calls 
                # the forward pass of the model. Needs further investigation...
                #xgen = self.model(cinn_input=z, cond_input=obs_rep,
                #                  rev=True, cut_ouput=cut_ouput)

                # use the cond_input calculated before the loop
                cond_input_rep = [torch.repeat_interleave(c,num_samples,dim = 0) for c in cond_input]
     
                xgen, _ = self.model.cinn(z, c=cond_input_rep, rev=True)

                xmean = xmean + torch.sum(xgen, dim=0, keepdim=True)
      
                if return_std:
                    xgen_list.append(xgen.cpu())
                    
            # free some memory
            #del z, obs_rep, xgen
            del z, cond_input_rep, xgen 
            
            xmean = xmean / self.samples_per_reco
                    
            if return_std:
                xgen = torch.cat(xgen_list, axis=0)
                xstd = torch.std(xgen, dim=0, keepdim=True).to(self.model.device)
            
            if not return_torch:
                xmean = np.squeeze(xmean.detach().cpu().numpy())
                xmean = self.reco_space.element(xmean)
                if return_std:
                    xstd = np.squeeze(xstd.detach().cpu().numpy())
                    xstd = self.reco_space.element(xstd)
        
            if return_std:
                return xmean, xstd
            else:
                return xmean


    def num_train_params(self):
        params_trainable = list(filter(lambda p: p.requires_grad,
                                        self.model.parameters()))

        print("Number of trainable params: ",
              sum(p.numel() for p in params_trainable))
        
    def load_learned_params(self, path, checkpoint:bool = True,
                            strict:bool = False):
        """
        Load a model from the given path. To load a model along with its
        weights, biases and module_arguments use a checkpoint.

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        checkpoint : bool, optional
            DESCRIPTION. The default is True.
        strict : bool, optional
            Strict loading of all parameters. Will raise an error if there
            are unknown weights in the file.
            The default is False

        Returns
        -------
        None.

        """
        if checkpoint:
            print("Load from ", path)
            # The operator cannot be stored in the checkpoint file. Therefore,
            # we have to provide him separately.
            self.model = CINN.load_from_checkpoint(path,
                                                   strict=strict,
                                                   operator=self.op)
            
            # Update the hyperparams of the reconstructor based on the hyper-
            # params of the model. Hyperparams for the optimizer and training
            # routine are ignored.
            hparams = self.model.hparams
            
            # set regular hyperparams
            self.img_size = hparams.img_size
            self.downsample_levels = hparams.downsample_levels
            self.downsampling = hparams.downsampling
            self.sample_distribution = hparams.sample_distribution



