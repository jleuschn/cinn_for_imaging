"""
Invertible Unet model.
"""

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pytorch_lightning as pl

from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.reconstructors.networks.cond_net import ResBlock, CondUNet
from cinn_for_imaging.reconstructors.networks.layers import (
        InvertibleDownsampling, InvertibleUpsampling, FBPModule,
        Fixed1x1ConvOrthogonal)


class IUNet(pl.LightningModule):
    """
    PyTorch iUnet architecture for low-dose CT reconstruction.

    Attributes
    ----------
    iunet : torch module list
        Building blocks of the invertible network.
    cond_net : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.

    """

    def __init__(self, in_ch: int, img_size: tuple, operator,
                 sample_distribution: str = 'normal', conditioning: str = 'fbp',
                 conditional_args: dict = {'filter_type': 'Hann',
                                           'frequency_scaling': 1.,
                                           'train_cond': False},
                 optimizer_args: dict = {'lr': 0.001, 'weight_decay': 0.},
                 downsample_levels: int = 5, clamping: float = 1.5,
                 downsampling: str = 'standard',
                 coupling: str = 'affine',
                 train_noise: tuple = (0, 0),
                 depth: int = 1,
                 data_range: list = [0, 1],
                 clamp_all: bool = False,
                 permute: bool = False,
                 special_init: bool = True,
                 normalize_inn: bool = True,
                 permute_type: str = 'random',
                 **kwargs):
        """
        IUNet constructor.

        Parameters
        ----------
        in_ch : int
            Number of input channels. This should be 1 for regular CT.
        img_size : tuple of int
            Size (h,w) of the input image.
        operator : Type
            Forward operator. This is the ray transform for CT.
        sample_distribution: str, optional
            Distribution Z of the random samples z.
            The default is 'normal'
        conditioning : str, optional
            Type of the conditioning network H. Currently, the only supported
            option is 'fpb'.
            The default is 'fbp'.
        conditional_args : dict, optional
            Arguments for the conditional network.
            The default are options for the FBP conditioner: filter_type='hann'
            and frequency_scaling=1.
        optimizer_args : dict, optional
            Arguments for the optimizer.
            The defaults are lr=0.001 and weight_decay=1e-5 for the ADAM
            optimizer.
        downsample_levels : int, optional
            Number of 1/2 downsampling steps in the network. This option is 
            currently not in active use for the structure of the network!
            The default is 5.
        clamping : float, optional
            The default is 1.5.
        downsampling : str, optional
            Type of the downsampling layers. Options are
                'standard' (same as 'invertible'), 'haar' and 'reshape'.
            The default is 'standard'.
        permute_type: str, optional 
                'random': random shuffling of channels/dimensions
                '1x1': Fixed 1x1 orhtogonal convolution
            The default is 'random'.
        Returns
        -------
        None.

        """
        super().__init__()

        # all inputs to init() will be stored (if possible) in a .yml file
        # alongside the model. You can access them via self.hparams.
        self.save_hyperparameters()

        # shorten some of the names or store values that can't be placed in
        # a .yml file
        self.in_ch = self.hparams.in_ch
        self.img_size = self.hparams.img_size
        self.downsample_levels = self.hparams.downsample_levels
        self.coupling = self.hparams.coupling
        self.train_noise = train_noise
        self.op = operator
        self.depth = depth
        self.data_range = data_range

        # choose the correct loss function
        self.criterion = CINNNLLLoss(
            distribution=self.hparams.sample_distribution)

        self.train_cond = conditional_args['train_cond']

        # set the list of downsamling layers
        if self.hparams.downsampling == 'standard':
            self.ds_list = ['invertible'] * self.hparams.downsample_levels
        else:
            self.ds_list = [downsampling] * self.hparams.downsample_levels

        # initialize the input padding layer
        pad_size = (self.img_size[0] - self.op.domain.shape[0],
                    self.img_size[1] - self.op.domain.shape[1])
        self.img_padding = torch.nn.ReflectionPad2d((0, pad_size[0],
                                                     0, pad_size[1]))

        # build the iUnet
        self.cinn = self.build_iunet()

        # choose the conditioning network
        if self.hparams.conditioning == 'fbp':
            prep_layers = nn.Sequential(
                FBPModule(ray_trafo=self.op,
                          filter_type=conditional_args['filter_type'],
                          frequency_scaling=conditional_args['frequency_scaling']),
                torch.nn.ReflectionPad2d((0, pad_size[0], 0, pad_size[1]))
            )
        elif self.hparams.conditioning == 'fft':
            prep_layers = nn.Sequential(
                torch.nn.ReflectionPad2d((0, pad_size[0], 0, pad_size[1]))
            )
        else:
            raise NotImplementedError

        if self.data_range[0] == 0 and self.data_range[1] == 1:
            use_sigmoid = True
        else:
            use_sigmoid = False

        self.cond_net = CondUNet(prep_layers=prep_layers,
                                 downsample_levels=self.downsample_levels,
                                 use_sigmoid=use_sigmoid)

        # initialize the values of the parameters
        if self.hparams.special_init:
            self.init_params()

    def build_iunet(self):
        """
        Connect the building blocks of the iUnet.

        Returns
        -------
        FrEIA ReversibleGraphNet
            iUnet model.

        """
        splits = list()

        # Create list of all condittioning nodes
        conditions = list()
        conditions.append(Ff.ConditionNode(self.in_ch, self.img_size[0],
                                           self.img_size[1],
                                           name='cond_inp'))

        for i in range(self.downsample_levels-1):
            conditions.append(Ff.ConditionNode((2**(i+2))*self.in_ch,
                                               int(1/(2**(i+1)) *
                                                   self.img_size[0]),
                                               int(1/(2**(i+1)) *
                                                   self.img_size[1]),
                                               name='cond_down_' + str(i)))

        for i in reversed(range(self.downsample_levels-2)):
            conditions.append(Ff.ConditionNode((2**(i+2))*self.in_ch,
                                               int(1/(2**(i+1)) *
                                                   self.img_size[0]),
                                               int(1/(2**(i+1)) *
                                                   self.img_size[1]),
                                               name='cond_up_' + str(i)))

        # Create input node
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1],
                              name='inp')]

        # Normalization of the input
        if self.hparams.normalize_inn:
            nodes.append(
                Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="ActNorm_inp"))

        # Conditioning on input level - only used in the network if there
        # are more than one input channels.
        if self.in_ch > 1:
            _add_conditioned_section(nodes, depth=1, cond_level=0,
                                     conditions=conditions,
                                     coupling=self.coupling,
                                     normalize=self.hparams.normalize_inn,
                                     clamp_all=self.hparams.clamp_all,
                                     permute=self.hparams.permute, 
                                     permute_type=self.hparams.permute_type)

        # First coupling blocks without conditioning
        if self.depth > 0:
            _add_downsample(nodes, self.ds_list[0])
            _add_conditioned_section(nodes, depth=self.depth, cond_level=0,
                                     conditions=None, coupling=self.coupling,
                                     normalize=self.hparams.normalize_inn,
                                     clamp_all=self.hparams.clamp_all,
                                     permute=self.hparams.permute, 
                                     permute_type=self.hparams.permute_type)
            _add_upsample(nodes, self.ds_list[0])

        # Loop for the downsampling path
        for i in range(self.downsample_levels-1):
            _add_downsample(nodes, self.ds_list[i])

            _add_conditioned_section(nodes, depth=1, cond_level=i+1,
                                     conditions=conditions,
                                     coupling=self.coupling,
                                     normalize=self.hparams.normalize_inn,
                                     clamp_all=self.hparams.clamp_all,
                                     permute=self.hparams.permute, 
                                     permute_type=self.hparams.permute_type)

            if self.depth > 0:
                _add_conditioned_section(nodes, depth=self.depth, cond_level=0,
                                         conditions=None,
                                         coupling=self.coupling,
                                         normalize=self.hparams.normalize_inn,
                                         clamp_all=self.hparams.clamp_all,
                                         permute=self.hparams.permute, 
                                         permute_type=self.hparams.permute_type)

            if i < (self.downsample_levels - 2):
                nodes.append(Ff.Node(nodes[-1], Fm.Split,
                                     {'n_sections': 2, 'dim': 0},
                                     name="split_" + str(i)))

                splits.append(nodes[-1].out1)

        # Loop for the upsampling path
        for i in reversed(range(self.downsample_levels-2)):
            _add_upsample(nodes, self.ds_list[i])

            nodes.append(Ff.Node([splits[i]] + [nodes[-1].out0],
                                 Fm.Concat, {'dim': 0}, name='concat_splits_' + str(i)))

            _add_conditioned_section(nodes, depth=1,
                                     cond_level=2*(self.downsample_levels - 2) - i + 1,
                                     conditions=conditions, coupling=self.coupling,
                                     normalize=self.hparams.normalize_inn,
                                     clamp_all=self.hparams.clamp_all,
                                     permute=self.hparams.permute, 
                                     permute_type=self.hparams.permute_type)

            if self.depth > 0:
                _add_conditioned_section(nodes, depth=self.depth, cond_level=0,
                                         conditions=None, coupling=self.coupling,
                                         normalize=self.hparams.normalize_inn,
                                         clamp_all=self.hparams.clamp_all,
                                         permute=self.hparams.permute, 
                                         permute_type=self.hparams.permute_type)

        # Final block
        _add_conditioned_section(nodes, depth=1, cond_level=0,
                                 conditions=None, coupling=self.coupling,
                                 normalize=False,
                                 clamp_all=self.hparams.clamp_all,
                                 permute=self.hparams.permute, 
                                 permute_type=self.hparams.permute_type)

        _add_upsample(nodes, self.ds_list[0])

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='flatten'))

        nodes.append(Ff.OutputNode(nodes[-1], name='out'))

        return Ff.ReversibleGraphNet(nodes + conditions, verbose=False)

    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
        for key, param in self.cinn.named_parameters():
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)

    def forward(self, cinn_input, cond_input, rev: bool = True,
                cut_ouput: bool = True):
        """
        Inference part of the whole model. There are two directions of the
        iUnet. These are controlled by rev:
            rev==True:  Create a reconstruction x for a random sample z
                        and the conditional measurement y (Z|Y) -> X.
            rev==False: Create a sample z from a reconstruction x
                        and the conditional measurement y (X|Y) -> Z .

        Parameters
        ----------
        cinn_input : torch tensor
            Input to the iUnet model. Depends on rev:
                rev==True: Random vector z.
                rev== False: Reconstruction x.
        cond_input : torch tensor
            Input to the conditional network. This is the measurement y.
        rev : bool, optional
            Direction of the iUnet flow. For True it is Z -> X to create a 
            single reconstruction. Otherwise X -> Z.
            The default is True.
        cut_ouput : bool, optional
            Cut the output of the network to the domain size of the operator.
            This is only relevant if rev==True.
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
            x, _ = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev,
                             jac=False)

            if cut_ouput:
                return x[:, :, :self.op.domain.shape[0], :self.op.domain.shape[1]]
            else:
                return x
        # direction (X|Y) -> Z
        else:
            cinn_input = self.img_padding(cinn_input)
            z, log_jac = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev,
                                   jac=True)
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

        # pad gt image to the right size
        cinn_input = self.img_padding(cinn_input)

        # run the conditional network
        c = self.cond_net(y)

        # run the iUnet from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(cinn_input, c, jac=True)

        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)

        if self.train_cond > 0:
            cond = self.cond_net(y)
            reco = cond[0]
            mse = torch.nn.functional.mse_loss(
                reco[:, :, :self.op.domain.shape[0], :self.op.domain.shape[1]],
                gt[:, :, :self.op.domain.shape[0], :self.op.domain.shape[1]])
            loss = loss + self.train_cond*mse
            self.log('train_cond_mse', mse)

        # Log the training loss
        self.log('train_loss', loss)

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

        # pad gt image to the right size
        gt = self.img_padding(gt)

        # run the conditional network
        c = self.cond_net(y)

        # run the iUnet from X -> Z with the gt data and conditioning
        zz, log_jac = self.cinn(gt, c, jac=True)

        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)

        if self.train_cond > 0.:
            cond = self.cond_net(y)
            reco = cond[0]
            mse = torch.nn.functional.mse_loss(
                reco[:, :, :self.op.domain.shape[0], :self.op.domain.shape[1]],
                gt[:, :, :self.op.domain.shape[0], :self.op.domain.shape[1]])
            loss = loss + self.train_cond*mse
            self.log('val_cond_mse', mse)

        # checkpoint the model and log the loss
        self.log('val_loss', loss)

        if batch_idx == 0:
            self.last_batch = batch

        return loss

    def validation_epoch_end(self, result):
        y = self.last_batch[0]
        gt = self.last_batch[1]
        img_grid = torchvision.utils.make_grid(
            gt, normalize=True, scale_each=True)
        self.logger.experiment.add_image("1) Ground Truth", img_grid,
                                         global_step=self.current_epoch)

        z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                        device=self.device)

        # Draw random samples from the radial distribution if needed
        if self.hparams.sample_distribution == 'radial':
            z_norm = torch.norm(z, dim=1)
            r = torch.abs(torch.randn((gt.shape[0], 1), device=self.device))
            z = z/z_norm.view(-1,1)*r

        with torch.no_grad():
            prep = self.cond_net.prep_layers(y)
            c = self.cond_net(y)[0][:, :, :self.op.domain.shape[0],
                                    :self.op.domain.shape[1]]

            x = self.forward(z, y, rev=True, cut_ouput=True)

            reco_grid = torchvision.utils.make_grid(
                x, normalize=True, scale_each=True)
            self.logger.experiment.add_image("2) Reconstructions",
                                             reco_grid, global_step=self.current_epoch)

            cond_reco_grid = torchvision.utils.make_grid(
                c, normalize=True, scale_each=True)
            self.logger.experiment.add_image("3) Conditional Reconstructions",
                                             cond_reco_grid, global_step=self.current_epoch)

            prep_grid = torchvision.utils.make_grid(
                prep, normalize=True, scale_each=True)
            self.logger.experiment.add_image("4) Classical Reconstructions",
                                             prep_grid, global_step=self.current_epoch)

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

        sched_factor = 0.999  # new_lr = lr * factor
        sched_patience = 5
        sched_trehsh = 0.005
        sched_cooldown = 1

        reduce_on_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=sched_factor, patience=sched_patience,
            threshold=sched_trehsh, min_lr=1e-07, eps=1e-08,
            cooldown=sched_cooldown, verbose=False)

        schedulers = {
            'scheduler': reduce_on_plateu,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1}

        return [optimizer], [schedulers]


def random_orthog(n):
    """
    Create a random, orthogonal n x n matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    torch float tensor
        Orthogonal matrix.

    """
    w = np.random.randn(n, n)
    w = w + w.T
    w, _, _ = np.linalg.svd(w)
    return torch.FloatTensor(w)


def subnet(in_ch, out_ch):
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
    return(nn.Sequential(ResBlock(in_ch, in_ch),
                         ResBlock(in_ch, out_ch),
                         nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1)))


def subnet_tanh(in_ch, out_ch):
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
    return(nn.Sequential(subnet(in_ch, out_ch),
                         nn.Tanh()))


def _add_conditioned_section(nodes, depth:int, cond_level:int, conditions:list, coupling:str,
                             clamping:int = 1.5, normalize:bool = True, clamp_all:bool = False,
                             permute:bool = False, permute_type: str = 'random'):
    """
    Add conditioned notes to the network.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    depth : int
        Number of layer units in this block.
    cond_level : int
        Current conditioning level.
    conditions : TYPE
        List of FrEIA condition notes.

    Returns
    -------
    None.

    """

    if not conditions == None:
        conditions = conditions[cond_level]

    for k in range(depth):

        # Choose correct sub-network and clamping function
        if clamp_all and not coupling == 'allInOne':
            subnetwork = subnet_tanh
            clamp_activation = nn.Identity()
        else:
            subnetwork = subnet
            clamp_activation = 'TANH'

        # Add permutation (if needed)
        if permute and not coupling == 'allInOne':
            if permute_type == '1x1':
                nodes.append(Ff.Node(nodes[-1].out0, Fixed1x1ConvOrthogonal, {},
                                 name="Permute_{}_{}".format(cond_level, k)))
            else:
                nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {},
                                 name="Permute_{}_{}".format(cond_level, k)))
        # Add coupling block
        if coupling == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnetwork,
                                  'clamp': clamping,
                                  'clamp_activation': clamp_activation},
                                 conditions=conditions,
                                 name="GLOWBlock_{}_{}".format(cond_level, k)))

        if coupling == 'RNVP':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.RNVPCouplingBlock,
                                 {'subnet_constructor': subnetwork,
                                  'clamp': clamping,
                                  'clamp_activation': clamp_activation},
                                 conditions=conditions,
                                 name="RNVPlock_{}_{}".format(cond_level, k)))

        if coupling == 'allInOne':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.AllInOneBlock,
                                 {'subnet_constructor': subnetwork,
                                  'affine_clamping': clamping,
                                  'gin_block': True,
                                  'global_affine_init': 1.0,
                                  'global_affine_type': 'SOFTPLUS',
                                  'permute_soft': False,
                                  'learned_householder_permutation': 0,
                                  'reverse_permutation': False},
                                 conditions=conditions,
                                 name="AllInOneBlock_{}_{}".format(cond_level, k)))

        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                                 {'subnet_constructor': subnetwork},
                                 conditions=conditions,
                                 name="NICEBlock_{}_{}".format(cond_level, k)))

        # Add normalization (if needed)
        if normalize and not coupling == 'allInOne':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {},
                                 name="ActNorm_{}_{}".format(cond_level, k)))


def _add_downsample(nodes, downsample, clamping=1.5):
    """
    Downsampling operations.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsample : str
        Downsampling method. Currently there are three options: 'haar', 
        'reshape' and 'invertible'.
    clamping : float, optional
        The default value is 1.5.

    Returns
    -------
    None.

    """

    if downsample == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling,
                             {'rebalance': 0.5, 'order_by_wavelet': True},
                             name='haar_down'))

    if downsample == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetDownsampling, {},
                             name='reshape_down'))

    if downsample == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleDownsampling,
                             {'stride': 2, 'method': 'cayley', 'init': 'haar',
                              'learnable': True}, name='invertible_down'))


def _add_upsample(nodes, upsample, clamping=1.5):
    """
    upsampling operations.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    upsample : str
        Upsamplig method. Currently there are two options: 'haar' and 
        'reshape'.
    clamping : float, optional
        The default value is 1.5.

    Returns
    -------
    None.

    """

    if upsample == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarUpsampling, {},
                             name='haar_up'))

    if upsample == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetUpsampling, {},
                             name='reshape_up'))

    if upsample == 'invertible':
        nodes.append(Ff.Node(nodes[-1].out0, InvertibleUpsampling,
                             {'stride': 2, 'method': 'cayley', 'init': 'haar',
                              'learnable': True}, name='invertible_up'))
