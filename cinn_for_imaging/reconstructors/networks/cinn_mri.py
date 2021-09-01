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

from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.reconstructors.networks.layers import InvertibleDownsampling, Fixed1x1ConvOrthogonal, Split
from cinn_for_imaging.reconstructors.networks.cond_net_mri import AvgPoolCondNetFBP, SimpleCondNetFBP, ResNetCondNet

from dival.reconstructors.networks.unet import UNet

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
                 optimizer_args: dict = {'lr': 0.001, 'weight_decay': 0.0},
                 conditioning: str ='ResNetCond',
                 downsample_levels: int = 6,
                 downsampling: str = 'invertible',
                 sample_distribution: str = 'normal',
                 coupling: str = 'affine',
                 use_fc_block: bool = True,
                 use_act_norm: bool = False, 
                 permutation: str = '1x1',
                 cond_conv_channels = [4, 16, 32, 64, 64, 32], 
                 cond_fc_size: int = 64, 
                 num_blocks: int = 6, 
                 num_fc_blocks: int = 2,
                 train_noise = (0.,0.),
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
        conditioning : str, optional
            Type of the conditioning network H. 
                'ResNetCond': ResNet backbone and ResNet heads
                'SimpleCondNet': ResNet backbone and Downsampling heads
                'AvgPoolCondNet': Just Downsampling heads
            The default is 'ResNetCond'.
        downsample_levels: int, optional
            Number of downsampling levels. Image size must be divisible by 2**downsample_levels
        downsampling: str, optional
            Type of downsampling: 
                'invertible' : learnable downsampling operation by Etmann et. al. 
                'haar' : downsampling based on haar wavelets
                'checkerboard' : standard iRevNet downsampling 
        sample_distribution: str, optional
            Distribution for latent variables z. Is used to define the maximum likelihood objective.
                'normal' : Gaussian Normaldistribution
                'radial': radial distribution (normal pushed to origin)
        coupling: str, optional
            Type of coupling. 
                'affine': GlowCouplingBlocks
                'additive': NICECouplingBlocks
        use_fc_block:
            Whether to use a last fully connected coupling block. 
        use_act_norm:
            Whether to use ActNorm after coupling layers. 
        permutation:
            Type of permutation to use after coupling blocks
                '1x1' : Fixed1x1 Convolution 
                'PermuteRandom: Random channel wise permutation
        cond_conv_channels:
            Channels for conditional input in each downsampling step. 
        cond_fc_size:
            size of conditional input for fully connected block 
        num_blocks: 
            number of coupling, permute, act norm blocks 
        num_fc_blocks:
            number of fully connected blocks
        train_noise: optional
            mean and variance of train noise 
        Returns
        -------
        None.

        """
        super().__init__()
        
        assert coupling in ['affine', 'additive'], 'coupling can either be affine or additive'
        assert downsampling in ['invertible', 'haar', 'checkerboard'], 'downsampling has to be invertible, haar or checkerboard'
        assert sample_distribution in ['normal', 'radial'], 'sample_distribution has to be normal or radial'
        assert conditioning in ["ResNetCond", "SimpleCondNet", "AvgPoolCondNet"], "conditioning {} not implemented.".format(conditioning)


        # all inputs to init() will be stored (if possible) in a .yml file 
        # alongside the model. You can access them via self.hparams.
        self.save_hyperparameters()
        
        # shorten some of the names or store values that can't be placed in
        # a .yml file
        self.in_ch = self.hparams.in_ch
        self.img_size = self.hparams.img_size
        self.permutation = permutation
        self.downsample_levels = self.hparams.downsample_levels
        self.downsampling = self.hparams.downsampling 
        self.coupling = self.hparams.coupling 
        self.sample_distribution = self.hparams.sample_distribution
        self.use_fc_block = self.hparams.use_fc_block
        self.use_act_norm = self.hparams.use_act_norm
        self.cond_conv_channels = self.hparams.cond_conv_channels
        self.cond_fc_size = self.hparams.cond_fc_size
        self.num_blocks = self.hparams.num_blocks        
        self.num_fc_blocks = self.hparams.num_fc_blocks
        self.train_noise = train_noise
        self.data_range = None
        # choose the correct loss function
        self.criterion = CINNNLLLoss(
                            distribution=self.hparams.sample_distribution)
        
        # build the cINN
        self.cinn = self.build_inn()
        
        # choose the conditioning network
        if self.hparams.conditioning == 'ResNetCond':
            self.cond_net = ResNetCondNet(
                             img_size=self.img_size,
                             downsample_levels=self.hparams.downsample_levels,
                             cond_conv_channels=self.cond_conv_channels, 
                             use_fc_block = self.use_fc_block, 
                             cond_fc_size=self.cond_fc_size)
        elif self.hparams.conditioning == "SimpleCondNet":
            self.cond_net = SimpleCondNetFBP(
                             img_size=self.img_size,
                             downsample_levels=self.hparams.downsample_levels,
                             cond_conv_channels=self.cond_conv_channels, 
                             use_fc_block = self.use_fc_block, 
                             cond_fc_size=self.cond_fc_size)
        elif self.hparams.conditioning == "AvgPoolCondNet":
            self.cond_net = AvgPoolCondNetFBP(
                             img_size=self.img_size,
                             downsample_levels=self.hparams.downsample_levels,
                             cond_conv_channels=self.cond_conv_channels, 
                             use_fc_block = self.use_fc_block, 
                             cond_fc_size=self.cond_fc_size)
        else:
            raise NotImplementedError
        
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
        split_nodes = []
        conditions = []
        for i in range(self.downsample_levels):
            conditions.append(Ff.ConditionNode(self.cond_conv_channels[i], 
                            self.img_size[0]/(2**(i+1)), self.img_size[1]/(2**(i+1)), name="cond_{}".format(i)))
        if self.use_fc_block:
            conditions.append(Ff.ConditionNode(self.cond_fc_size, name="cond_{}".format(self.downsample_levels)))
        

        ### build the network & add conditioning and splits ###
        
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1],
                              name='inp')]
        # 1x320x320 -> 4x160x160
        _add_downsample(nodes, self.downsampling)

        for downsample_step in range(self.downsample_levels -1 ):
            _add_conditioned_section(nodes, 
                                    downsampling_level=downsample_step,
                                    cond=conditions[downsample_step],
                                    num_blocks=self.num_blocks,
                                    coupling=self.coupling,
                                    act_norm=self.use_act_norm,
                                    permutation=self.permutation)

            _add_downsample(nodes, self.downsampling)


            nodes.append(Ff.Node(nodes[-1], Split,
                            {'n_sections': 2, 'dim': 0}, 
                            # {'section_sizes':[8*2**downsample_step,8*2**downsample_step], 'dim':0},
                            name="split_{}".format(downsample_step)))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                   name='flatten_split_{}'.format(downsample_step)))


        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {},
                            name='flatten'))

        if self.use_fc_block:
            nodes.append(Ff.Node(nodes[-1], Split,
                            {'section_sizes': [128], 'dim' : 0, 'n_sections': None}, 
                            name="split_fc"))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                   name='flatten_split_fc'))

            _add_fc_section(nodes, 
                            cond=conditions[-1], 
                            num_blocks=self.num_fc_blocks, 
                            coupling=self.coupling)                    


        ## 5) concat all split notes and network output
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}, name='concat_splits'))
  
        nodes.append(Ff.OutputNode(nodes[-1], name='out'))
        
        return Ff.GraphINN(nodes + conditions + split_nodes,
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
            x, log_jac = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev, jac=False)
            return x, log_jac
        # direction (X|Y) -> Z
        else:
            z, log_jac = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)
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
        y = batch[0]
        gt = batch[1]

        # run the conditional network
        c = self.cond_net(y)
        
        # Sample noise from Normal(train_noise[0], train_noise[1])
        if self.train_noise[1] > 0:
            cinn_input = gt + torch.randn((gt.shape),
                                          device=self.device)*self.train_noise[1] + self.train_noise[0]
            
            if self.data_range is not None:
                cinn_input = torch.clip(cinn_input, 
                                        min=self.data_range[0],
                                        max=self.data_range[1])
        else:
            cinn_input = gt


        # run the cINN from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(cinn_input, c)

        
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
        y = batch[0]
        gt = batch[1]
                
        # run the conditional network
        c = self.cond_net(y)
        
        # run the cINN from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(gt, c)
        
        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        
        # checkpoint the model and log the loss
        self.log('val_loss', loss)

        return loss
    
    def training_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)

        y = self.last_batch[0]
        gt = self.last_batch[1]
        img_grid = torchvision.utils.make_grid(gt,scale_each=True,normalize=True)
        self.logger.experiment.add_image("ground truth",
                    img_grid, global_step=self.current_epoch)
        
        z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                        device=self.device)
        # Draw random samples from the radial distribution if needed
        if self.hparams.sample_distribution == 'radial':
            z_norm = torch.norm(z, dim=1)
            r = torch.abs(torch.randn((gt.shape[0], 1), device=self.device))
            z = z/z_norm.view(-1,1)*r
        with torch.no_grad():

            x, _ = self.forward(z, y, rev=True)
            
            reco_grid = torchvision.utils.make_grid(x,scale_each=True,normalize=True)
            self.logger.experiment.add_image("reconstructions",
                    reco_grid, global_step=self.current_epoch)

            y_grid = torchvision.utils.make_grid(y, scale_each=True,normalize=True)
            self.logger.experiment.add_image("zero filled ifft",
                    y_grid, global_step=self.current_epoch)

            conds = self.cond_net(y)
            for i, c in enumerate(conds): 
                c = c.view(-1, 1, c.shape[-2], c.shape[-1])
                c_grid = torchvision.utils.make_grid(c, scale_each=True,normalize=True)
                self.logger.experiment.add_image("cond_level_{}".format(i),
                    c_grid, global_step=self.current_epoch)


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
        
        sched_factor = 0.9 # new_lr = lr * factor
        sched_patience = 4 
        sched_tresh = 0.005
        sched_cooldown = 1

        reduce_on_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, factor=sched_factor, 
                            patience=sched_patience, threshold=sched_tresh,
                            min_lr=1e-9, eps=1e-08, cooldown=sched_cooldown,
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
                nn.Conv2d(in_ch, 2*in_ch, 3, padding=1), 
                nn.LeakyReLU(), 
                nn.Conv2d(2*in_ch, 2*in_ch, 3, padding=1), 
                nn.LeakyReLU(),
                nn.Conv2d(2*in_ch, out_ch, 3, padding=1))


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
                nn.Conv2d(in_ch, 2*in_ch, 1), 
                nn.LeakyReLU(), 
                nn.Conv2d(2*in_ch, 2*in_ch, 1), 
                nn.LeakyReLU(),
                nn.Conv2d(2*in_ch, out_ch, 1))


def subnetUncond(in_ch, out_ch):
    """
    Sub-netwok with 1x1 2d-convolutions for unconditioned parts of the cINN.

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
                nn.Conv2d(in_ch, 2*in_ch, 1),
                nn.LeakyReLU(), 
                nn.Conv2d(2*in_ch, 2*in_ch, 1), 
                nn.LeakyReLU(),
                nn.Conv2d(2*in_ch, out_ch, 1))


def subnet_fc(in_ch, out_ch):
    """
    Sub-network with fully connected layers and leaky ReLU activation.

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
    return nn.Sequential(nn.Linear(in_ch, 2*in_ch), 
                         nn.LeakyReLU(), 
                         nn.Linear(2*in_ch, 2*in_ch),
                         nn.LeakyReLU(),
                         nn.Linear(2*in_ch, out_ch))    
    

def _add_downsample(nodes, downsample):
    """
    Downsampling operations.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsample : str
        Downsampling method. Currently there are three options: 'haar', 
        'reshape' and 'invertible'.

    Returns
    -------
    None.

    """
    
    if downsample == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, 
                               {'rebalance':0.5, 'order_by_wavelet':True},
                               name='haar'))
    if downsample == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetDownsampling, {},
                               name='reshape')) 
          
    if downsample == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleDownsampling,
                             {'stride':2, 'method':'cayley', 'init':'haar',
                              'learnable':True}, name='invertible')) 
    


def _add_conditioned_section(nodes, downsampling_level, num_blocks, cond, coupling, act_norm, permutation):
    """
    Add conditioned notes to the network.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsampling_level: int
        Current downsampling level
    num_blocks: int
        Number of coupling blocks    
    cond : TYPE
        FrEIA condition note
    coupling: str
        Type of coupling used 
    act_norm: bool
        whether to use act norm
    permutation: str
        which permutation to use
    Returns
    -------
    None.

    """
    
    for k in range(num_blocks):
        if k % 2 == 0:
            subnet = subnet_conv1x1
        else:
            subnet = subnet_conv3x3
        
        if coupling == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet, 'clamp':1.5},
                             conditions = cond,
                             name="GLOWBlock_{}.{}".format(downsampling_level, k)))
        else: 
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                             {'subnet_constructor':subnet},
                             conditions = cond,
                             name="NICEBlock_{}.{}".format(downsampling_level, k)))
        
        if act_norm:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="ActNorm_{}.{}".format(downsampling_level, k)))

        if permutation == "1x1":
            nodes.append(Ff.Node(nodes[-1].out0, Fixed1x1ConvOrthogonal, 
                                 {}, 
                                 name='1x1Conv_{}.{}'.format(downsampling_level, k)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, 
                                 {'seed':(k+1)*(downsampling_level+1)}, 
                                 name='PermuteRandom_{}.{}'.format(downsampling_level, k)))


def _add_fc_section(nodes, cond, num_blocks, coupling):
    """
    Add conditioned notes to the network.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    num_blocks: int
        Number of coupling blocks    
    cond : TYPE
        FrEIA condition note
    coupling: str
        Type of coupling used 
    Returns
    -------
    None.

    """
    
    for k in range(num_blocks):        
        if coupling == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_fc, 'clamp':1.5},
                             conditions = cond,
                             name="GLOWBlock_fc.{}".format(k)))
        else: 
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                             {'subnet_constructor':subnet_fc},
                             conditions = cond,
                             name="NICEBlock_fc.{}".format(k)))
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, 
                                 {'seed':k}, 
                                 name='PermuteRandom_fc.{}'.format(k)))