"""
Baseline cINN model from the Master thesis of Alexander Denker for the 
LoDoPaB-CT dataset.
"""

from cinn_for_imaging.reconstructors.networks.cinn import Flatten, _add_downsample
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import torchvision

import numpy as np
import pytorch_lightning as pl
from copy import deepcopy

from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.reconstructors.networks.layers import InvertibleDownsampling, Fixed1x1ConvOrthogonal, GradientDescentStep, InvertibleUpsampling


class CINN(pl.LightningModule):
    """
    PyTorch cINN architecture for low-dose CT reconstruction.
    
    Attributes
    ----------
    cinn : torch module list
        Building blocks of the conditional network.
    cond_net : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """

    def __init__(self, in_ch:int, img_size,
                 op, step_size,
                 optimizer_args: dict = {'lr': 0.001, 'weight_decay': 0.0},
                 sample_distribution: str = 'normal',
                 downsampling: str = 'invertible',
                 coupling: str = 'affine',
                 use_act_norm: bool = False, 
                 permutation: str = '1x1',
                 num_blocks: int = 4, 
                 upsamling: str = 'invertible',
                 unrolling_steps: int = 8,
                 **kwargs):
        """
        FastMRI constructor.

        Parameters
        ----------
        in_ch : int
            Number of input channels. This should be 1 for regular CT.
        img_size : tuple of int
            Size (h,w) of the input image.
        optimizer_args : dict, optional
            Arguments for the optimizer.
            The defaults are lr=0.001 and weight_decay=0 for the ADAM
            optimizer.
        downsampling: str, optional
            Type of downsampling: 
                'invertible' : learnable downsampling operation by Etmann et. al. 
                'haar' : downsampling based on haar wavelets
                'reshape' : standard iRevNet downsampling 
        sample_distribution: str, optional
            Distribution for latent variables z. Is used to define the maximum likelihood objective.
                'normal' : Gaussian Normaldistribution
                'radial': radial distribution (normal pushed to origin)
        coupling: str, optional
            Type of coupling. 
                'affine': GlowCouplingBlocks
                'additive': NICECouplingBlocks
        use_act_norm:
            Whether to use ActNorm after coupling layers. 
        permutation:
            Type of permutation to use after coupling blocks
                '1x1' : Fixed1x1 Convolution 
                'PermuteRandom: Random channel wise permutation
        num_blocks: 
            number of coupling, permute, act norm blocks 
        upsamling: str, optional
                'invertible' : learnable upsampling operation by Etmann et. al. 
                'haar' : upsampling based on haar wavelets
                'checkerboard' : standard iRevNet upsampling 
        unrolling_steps: int, optional
            number of unrolled gradient descent steps to be implemented
        Returns
        -------
        None.

        """
        super().__init__()
        
        assert coupling in ['affine', 'additive'], 'coupling can either be affine or additive'
        assert downsampling in ['invertible', 'haar', 'reshape'], 'downsampling has to be invertible, haar or checkerboard'
        assert upsamling in ['invertible', 'haar', 'reshape'], 'upsampling has to be invertible, haar or checkerboard'
        assert sample_distribution in ['normal', 'radial'], 'sample_distribution has to be normal or radial'


        # all inputs to init() will be stored (if possible) in a .yml file 
        # alongside the model. You can access them via self.hparams.
        self.save_hyperparameters()
        
        # 
        self.op = op#OperatorModule(op)
        self.step_size = step_size

        # shorten some of the names or store values that can't be placed in
        # a .yml file
        self.in_ch = self.hparams.in_ch
        self.img_size = self.hparams.img_size
        
        self.downsampling = self.hparams.downsampling 
        self.coupling = self.hparams.coupling 
        self.sample_distribution = self.hparams.sample_distribution
        self.use_act_norm = self.hparams.use_act_norm
        self.permutation = self.hparams.permutation
        self.num_blocks = self.hparams.num_blocks    
        self.upsampling = self.hparams.upsampling 
        self.unrolling_steps = self.hparams.unrolling_steps    
        # choose the correct loss function
        self.criterion = CINNNLLLoss(
                            distribution=self.hparams.sample_distribution)
        
        # build the cINN
        self.cinn = self.build_inn()
                
        # initialize the values of the parameters
        self.init_params()
        
      
    def build_inn(self):
        """
        Connect the building blocks of the cINN.

        Returns
        -------
        FrEIA ReversibleGraphNet
            cINN model.

        """
        
        # initialize lists for the split and conditioning notes
        conditions = [Ff.ConditionNode(1, 1000, 513, name="cond")]
        

        ### build the network & add conditioning and splits ###
        
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1],
                              name='inp')]
        

        for k in range(self.unrolling_steps):
            _add_section(nodes, unrolling_step=k, cond=conditions[0], op=self.op, step_size=self.step_size,
                        downsampling=self.downsampling, upsampling=self.upsampling, 
                        coupling=self.coupling, permutation=self.permutation, act_norm=self.use_act_norm,
                        num_blocks=self.num_blocks)

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {},
                            name='flatten'))

        nodes.append(Ff.OutputNode(nodes[-1], name='out'))
        
        return Ff.GraphINN(nodes + conditions,
                                     verbose=False)
    
    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
        # approx xavier
        #for p in self.cond_net.parameters():
        #    p.data = 0.02 * torch.randn_like(p) 
            
        for key, param in self.cinn.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                # last convolution in the coeff func
                if len(split) > 3 and split[3][-1] == '4': 
                    param.data.fill_(0.)
    
    def forward(self, cinn_input, cond_input, rev:bool = True):
        """
        Inference part of the whole model. There are two directions of the
        cINN. These are controlled by rev:
            rev==True:  Create a reconstruction x for a random sample z
                        and the conditional measurement y (Z|Y) -> X.
            rev==False: Create a sample z from a reconstruction x
                        and the conditional measurement y (X|Y) -> Z .

        Parameters
        ----------
        cinn_input : torch tensor
            Input to the cINN model. Depends on rev:
                rev==True: Random vector z.
                rev== False: Reconstruction x.
        cond_input : torch tensor
            Input to the conditional network. This is the measurement y.
        rev : bool, optional
            Direction of the cINN flow. For True it is Z -> X to create a 
            single reconstruction. Otherwise X -> Z.
            The default is True.

        Returns
        -------
        torch tensor or tuple of torch tensor
            rev==True:  x : Reconstruction
            rev==False: z : Sample from the target distribution
                        log_jac : log det of the Jacobian

        """
        # direction (Z|Y) -> X
        if rev:
            x, log_jac = self.cinn(cinn_input, c=[cond_input], rev=rev, jac=False)
            return x, log_jac
        # direction (X|Y) -> Z
        else:
            z, log_jac = self.cinn(cinn_input,c=[cond_input], rev=rev)
            return z, log_jac

    def training_step(self, batch, batch_idx):
        """
        Pytorch Lightning training step. Should be independent of forward() 
        according to the documentation. The loss value is logged.

        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        result : TYPE
            Result of the training step.

        """
        y, gt = batch

        # run the cINN from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(gt, [y])

        
        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        
        # Log the training loss
        self.log('train_loss', loss)
        
        self.last_batch = batch
        return loss 
    
    def validation_step(self, batch, batch_idx):
        """
        Pytorch Lightning validation step. Should be independent of forward() 
        according to the documentation. The loss value is logged and the
        best model according to the loss (lowest) checkpointed.

        Parameters
        ----------
        batch : tuple of tensor
            Batch of measurement y and ground truth reconstruction gt.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        result : TYPE
            Result of the validation step.

        """
        y, gt = batch
                
        # run the conditional network
        
        # run the cINN from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(gt, [y])
        
        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        
        # checkpoint the model and log the loss
        self.log('val_loss', loss)

        return loss
    
    def training_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)

        y, gt = self.last_batch
        img_grid = torchvision.utils.make_grid(gt,scale_each=True,normalize=True)
        self.logger.experiment.add_image("ground truth",
                    img_grid, global_step=self.current_epoch)
        
        z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                        device=self.device)
        with torch.no_grad():

            x, _ = self.forward(z, y, rev=True)
            
            reco_grid = torchvision.utils.make_grid(x,scale_each=True,normalize=True)
            self.logger.experiment.add_image("reconstructions",
                    reco_grid, global_step=self.current_epoch)

            y_grid = torchvision.utils.make_grid(y, scale_each=True,normalize=True)
            self.logger.experiment.add_image("sinogram",
                    y_grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        """
        Setup the optimizer. Currently, the ADAM optimizer is used.

        Returns
        -------
        optimizer : torch optimizer
            The Pytorch optimizer.

        """
        optimizer = torch.optim.Adam(self.parameters(),
                     lr=self.hparams.optimizer_args['lr'], 
                     weight_decay=self.hparams.optimizer_args['weight_decay'])
        
        sched_factor = 0.2 # new_lr = lr * factor
        sched_patience = 2 
        sched_tresh = 0.005
        sched_cooldown = 1

        reduce_on_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, factor=sched_factor, 
                            patience=sched_patience, threshold=sched_tresh,
                            min_lr=0, eps=1e-08, cooldown=sched_cooldown,
                            verbose = False)

        schedulers = {
         'scheduler': reduce_on_plateu,
         'monitor': 'val_loss', 
         'interval': 'epoch',
         'frequency': 1 }

        return [optimizer], [schedulers]



class Flatten(nn.Module):
    """
    Torch module for flattening an input.

    Methods
    -------
    forward(x)
        Compute the forward pass.
        
    """
    
    def __init__(self, *args):
        super().__init__()
    
    def forward(self, x):
        """
        Will just leave the channel dimension and combine all 
        following dimensions.

        Parameters
        ----------
        x : torch tensor
            Input for the flattening.

        Returns
        -------
        torch tensor
            Flattened torch tensor.

        """
        return x.view(x.shape[0], -1)
            

def subnet_conv3x3(in_ch, out_ch):
    """
    Sub-network with 3x3 2d-convolutions and leaky ReLU activation.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    torch sequential model
        The sub-network.

    """
    return nn.Sequential(
                nn.Conv2d(in_ch, 128, 3, padding=1), 
                nn.LeakyReLU(), 
                nn.Conv2d(128, 128, 3, padding=1), 
                nn.LeakyReLU(),
                nn.Conv2d(128, out_ch, 3, padding=1))


def subnet_conv1x1(in_ch, out_ch):
    """
    Sub-network with 1x1 2d-convolutions and leaky ReLU activation.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    torch sequential model
        The sub-network.

    """
    return nn.Sequential(
                nn.Conv2d(in_ch, 128, 1), 
                nn.LeakyReLU(), 
                nn.Conv2d(128, 128, 1), 
                nn.LeakyReLU(),
                nn.Conv2d(128, out_ch, 1))
    


    

def _add_section(nodes, unrolling_step,cond, op, step_size, 
                downsampling, upsampling, coupling, permutation, act_norm,num_blocks):

    nodes.append(Ff.Node(nodes[-1].out0, GradientDescentStep, 
                            {'op':op , 'step_size':step_size }, 
                            conditions=cond, name="GradientDescent_{}".format(unrolling_step)))


    if downsampling == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, 
                               {'rebalance':0.5, 'order_by_wavelet':True},
                               name='haar'))
    if downsampling == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetDownsampling, {},
                               name='reshape')) 
          
    if downsampling == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleDownsampling,
                             {'stride':2, 'method':'cayley', 'init':'haar',
                              'learnable':True}, name='invertible')) 


    for k in range(num_blocks):
        if k % 2 == 0:
            subnet = subnet_conv1x1
        else:
            subnet = subnet_conv3x3
        
        if coupling == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet, 'clamp':1.5},
                             #conditions = cond,
                             name="GLOWBlock_{}.{}".format(unrolling_step, k)))
        else: 
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                             {'subnet_constructor':subnet},
                             #conditions = cond,
                             name="NICEBlock_{}.{}".format(unrolling_step, k)))
        
        if act_norm:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="ActNorm_{}.{}".format(unrolling_step, k)))

        if permutation == "1x1":
            nodes.append(Ff.Node(nodes[-1].out0, Fixed1x1ConvOrthogonal, 
                                 {}, 
                                 name='1x1Conv_{}.{}'.format(unrolling_step, k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, 
                                 {'seed':(k+1)*(unrolling_step+1)}, 
                                 name='PermuteRandom_{}.{}'.format(unrolling_step, k)))



    if upsampling == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarUpsampling, 
                               {},
                               name='haar'))
    if upsampling == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetUpsampling, {},
                               name='reshape')) 

    if upsampling == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleUpsampling,
                             {'stride': 2, 'method': 'cayley', 'init': 'haar',
                              'learnable': True}, name='invertible_up'))

