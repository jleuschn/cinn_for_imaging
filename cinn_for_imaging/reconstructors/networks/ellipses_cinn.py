"""
Baseline cINN model from the Master thesis of Alexander Denker for the 
LoDoPaB-CT dataset.
"""

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import pytorch_lightning as pl

from cinn_for_imaging.util.torch_losses import CINNNLLLoss
from cinn_for_imaging.reconstructors.networks.layers import NICECouplingBlock, InvertibleDownsampling, FBPModule


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

    def __init__(self, in_ch:int, img_size, operator, 
                 sample_distribution:str = 'normal', conditioning: str ='fbp',
                 conditional_args: dict = {'filter_type':'Hann',
                                           'frequency_scaling': 1.,},
                 optimizer_args: dict = {'lr': 0.001, 'weight_decay': 1e-5},
                 downsample_levels: int = 5, clamping: float = 1.5,
                 downsampling: str ='standard',
                 coupling: str ='affine',
                 **kwargs):
        """
        LowDoseCINN constructor.

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
            options are 'fbp' and 'lpd'. (lpd uses a chached dataset and is only supported for lodopab)
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
            Type of the downsampling layers. Options are:
                'standard': All reshape + last downsample is Haar
                'standard_learned': All reshape + last downsample learned
                                    invertible downsampling
                'reshape': Only reshape downsampling
                'haar': Only Haar downsampling
                'invertible': Only learned invertible downsampling
            The default is 'standard'.
        

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
        self.op = operator
        
        self.fc_cond_dim = int(256 * self.in_ch * \
            (int(self.img_size[0] / 2**(self.downsample_levels+1) / 5) * \
             int(self.img_size[1] / 2**(self.downsample_levels+1) / 5)) )
        
        # choose the correct loss function
        self.criterion = CINNNLLLoss(
                            distribution=self.hparams.sample_distribution)
        
        # set the list of downsamling layers
        if self.hparams.downsampling == 'standard':
            self.ds_list = ['reshape'] * (self.hparams.downsample_levels - 1)
            self.ds_list.append('haar')
        elif self.hparams.downsampling == 'standard_learned':
            self.ds_list = ['reshape'] * (self.hparams.downsample_levels - 1)
            self.ds_list.append('invertible')
        else:
            self.ds_list = [downsampling] * self.hparams.downsample_levels     
        
        # initialize the input padding layer
        pad_size = (self.img_size[0] - self.op.domain.shape[0],
                    self.img_size[1] - self.op.domain.shape[1]) 
        self.img_padding = torch.nn.ReflectionPad2d((0, pad_size[0],
                                                     0, pad_size[1]))
        
        # build the cINN
        self.cinn = self.build_inn()
        
        # choose the conditioning network
        if self.hparams.conditioning == 'fbp':
            self.cond_net = CondNetFBP(ray_trafo=self.op, 
                                       img_size=self.img_size,
                                       downsample_levels=3,
                                       **conditional_args)
        elif self.hparams.conditioning == 'lpd':
            self.cond_net = CondNet(ray_trafo=self.op,img_size=self.img_size,
                                       downsample_levels=self.hparams.downsample_levels)
        
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
        
        nodes = [Ff.InputNode(self.in_ch, self.img_size[0], self.img_size[1],
                              name='inp')]
        

        
        conditions = [Ff.ConditionNode(4*self.in_ch, # 4
                                       int(1/2*self.img_size[0]),
                                       int(1/2*self.img_size[1]), 
                                       name='cond_1'),
                      Ff.ConditionNode(8*self.in_ch, # 16
                                       int(1/4*self.img_size[0]),
                                       int(1/4*self.img_size[1]), 
                                       name='cond_2'), 
                      Ff.ConditionNode(16*self.in_ch, # 32
                                       int(1/8*self.img_size[0]),
                                       int(1/8*self.img_size[1]),
                                       name='cond_3')]
        
        split_nodes = []
        
        # 1 x 128 x 128 -> 4 x 64 x 64  (1/2)
        _add_downsample(nodes, 'invertible', in_ch=self.in_ch, coupling=self.coupling)

        # Condition level 0
        _add_conditioned_section(nodes, depth=6, in_ch=4*self.in_ch, 
                                 cond_level=0, conditions=conditions, coupling=self.coupling)

        # 4 x 64 x 64 -> 16 x 32 x 32 (1/4)
        _add_downsample(nodes, 'invertible', in_ch=4*self.in_ch, coupling=self.coupling)

        # Condition level 1
        _add_conditioned_section(nodes, depth=6, in_ch=16*self.in_ch, 
                                 cond_level=1, conditions=conditions, coupling=self.coupling)

        # 16 x 32 x 32 -> 64 x 16 x 16 (1/8)
        _add_downsample(nodes, 'invertible', in_ch=16*self.in_ch, coupling=self.coupling)
        
        # Split: each 32 x 16 x 16
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[32*self.in_ch,32*self.in_ch],
                              'dim':0}, name="split_1"))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {},
                                   name='flatten_split_1'))

        # Condition level 2
        _add_conditioned_section(nodes, depth=6, in_ch=32*self.in_ch, 
                                 cond_level=2, conditions=conditions, coupling=self.coupling)
        
        # 32 x 16 x 16 -> 128 x 8 x 8 (1/16)
        _add_downsample(nodes, 'invertible', in_ch=32*self.in_ch, coupling=self.coupling)

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='flatten'))       


        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}, name='concat_splits'))
  
        nodes.append(Ff.OutputNode(nodes[-1], name='out'))
        
        return Ff.ReversibleGraphNet(nodes + conditions + split_nodes,
                                     verbose=False)
    
    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
        # approx xavier
        for p in self.cond_net.parameters():
            p.data = 0.02 * torch.randn_like(p) 
            
        for key, param in self.cinn.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                if len(split) > 3 and split[3][-1] == '4': # last convolution in the coeff func
                    param.data.fill_(0.)
    
    def forward(self, cinn_input, cond_input, rev:bool = True,
                cut_ouput:bool = True):
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
            x = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)

            if cut_ouput:
                return x[:,:,:self.op.domain.shape[0],:self.op.domain.shape[1]]
            else:    
                return x
        # direction (X|Y) -> Z
        else:
            #print("Before padding:" ,cinn_input.shape)
            cinn_input = self.img_padding(cinn_input)
            #print("After padding: ", cinn_input.shape)
            z = self.cinn(cinn_input, c=self.cond_net(cond_input), rev=rev)
            log_jac = self.cinn.log_jacobian(run_forward=False)
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

        # pad gt image to the right size
        gt = self.img_padding(gt)

        # run the conditional network
        c = self.cond_net(y)
        
        # run the cINN from X -> Z with the gt data and conditioning
        zz = self.cinn(gt, c)
        log_jac = self.cinn.log_jacobian(run_forward=False)
        
        z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                        device=self.device)
        x_rec = self.cinn(z, c, rev=True, intermediate_outputs=False)

        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        #mse = nn.MSELoss()
        l2_loss = torch.sum((x_rec - gt) ** 2) / gt.shape[0]# mse(x_rec, gt)

        weight = 0.1
        # Log the training loss
        self.log('train_loss', loss + weight*l2_loss)
        self.log('train_nll', loss)
        self.log('train_mse_backward', l2_loss)
        self.last_batch = batch

        return loss + weight*l2_loss
    
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
        
        # pad gt image to the right size
        gt = self.img_padding(gt)
        
        # run the conditional network
        c = self.cond_net(y)
        
        # run the cINN from X -> Z with the gt data and conditioning
        zz = self.cinn(gt, c)
        log_jac = self.cinn.log_jacobian(run_forward=False)
        
        # evaluate the NLL loss
        loss = self.criterion(zz=zz, log_jac=log_jac)
        
        # checkpoint the model and log the loss
        self.log('val_loss', loss)

        return loss
    
    def training_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets too big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)

        y, gt = self.last_batch
        img_grid = torchvision.utils.make_grid(gt, normalize=True, scale_each=True)

        self.logger.experiment.add_image("ground truth",
                    img_grid, global_step=self.current_epoch)
        
        z = torch.randn((gt.shape[0], self.img_size[0]*self.img_size[1]),
                        device=self.device)

        with torch.no_grad():
            c = self.cond_net.forward(y)

            cond_level = 0
            for cond in c:
                cond = cond.view(-1, 1, cond.shape[-2], cond.shape[-1])
                cond_grid = torchvision.utils.make_grid(cond, normalize=True, scale_each=True)

                self.logger.experiment.add_image("cond_level_" + str(cond_level),
                    cond_grid, global_step=self.current_epoch)
                cond_level += 1

            x = self.forward(z, y, rev=True, cut_ouput=True)

            reco_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True)
            self.logger.experiment.add_image("reconstructions", reco_grid, global_step=self.current_epoch)

            if self.hparams.conditioning == 'fbp':
                fbp = self.cond_net.fbp_layer(y)
                fbp_grid = torchvision.utils.make_grid(fbp, normalize=True, scale_each=True)
                self.logger.experiment.add_image("filtered_back_projection", fbp_grid, global_step=self.current_epoch)

            elif self.hparams.conditioning == 'lpd':
                y_grid = torchvision.utils.make_grid(y, normalize=True, scale_each=True)
                self.logger.experiment.add_image("learnedpd_reco", y_grid, global_step=self.current_epoch)

    
   # def on_after_backward(self):
        # Called in the training loop after loss.backward() and before optimizers do anything.
       # if self.global_step % 25 == 0:
    #    total_norm = 0.0
    #    for name, param in self.named_parameters():
    #        if param.requires_grad:
    #            param_norm = param.grad.data.norm(2)
    #            total_norm += param_norm.item() ** 2
        
    #    total_norm = total_norm ** (1. / 2)
    #    print(total_norm)
            #self.logger.experiment.add_scalar("Train/GradNorm (on_after_backward)", total_norm, self.global_step)
    

    """
    def on_before_zero_grad(self, opt):
        #Called after optimizer.step() and before optimizer.zero_grad().
        if self.global_step % 25 == 0:
            total_norm = 0.0
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            
            total_norm = float(int(total_norm ** (1. / 2)))
            self.logger.experiment.add_scalar("Train/GradNorm (on_before_zero_grad)", total_norm, self.global_step)
    """

    def configure_optimizers(self):
        """
        Setup the optimizer. Currently, the ADAM optimizer is used.

        Returns
        -------
        optimizer : torch optimizer
            The Pytorch optimizer.

        """
        optimizer = torch.optim.Adam(self.parameters(),
                        lr=0.5*self.hparams.optimizer_args['lr'], 
                        weight_decay=0)#self.hparams.optimizer_args['weight_decay'])
        
        sched_factor = 0.4 # new_lr = lr * factor
        sched_patience = 2 
        sched_trehsh = 0.005
        sched_cooldown = 1

        reduce_on_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    factor=sched_factor,
                                                                    patience=sched_patience,
                                                                    threshold=sched_trehsh,
                                                                    min_lr=0, eps=1e-08,
                                                                    cooldown=sched_cooldown,
                                                                    verbose = False)

        schedulers = {
         'scheduler': reduce_on_plateu,
         'monitor': 'val_loss', 
         'interval': 'epoch',
         'frequency': 1 }

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


class CondNetFBP(nn.Module):
    """
    Conditional network H that sits on top of the invertible architecture. It 
    features a FBP operation at the beginning and continues with post-
    processing steps.
    
    Attributes
    ----------
    resolution_levels : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """
    
    def __init__(self, ray_trafo, img_size, downsample_levels=5,
                 filter_type='Hann', frequency_scaling=1.):
        """
        

        Parameters
        ----------
        ray_trafo : TYPE
            DESCRIPTION.
        img_size : TYPE
            DESCRIPTION.
        fc_cond_dim : int, optional
            DESCRIPTION.
        filter_type : TYPE, optional
            DESCRIPTION. The default is 'Hann'.
        frequency_scaling : TYPE, optional
            DESCRIPTION. The default is 1..

        Returns
        -------
        None.

        """
        super().__init__()

        # FBP and resizing layers
        self.fbp_layer = FBPModule(ray_trafo, filter_type=filter_type,
                 frequency_scaling=frequency_scaling)
        
        self.img_size = img_size
        self.dsl = downsample_levels

        pad_size = (img_size[0] - ray_trafo.domain.shape[0],
                    img_size[1] - ray_trafo.domain.shape[1]) 
        self.img_padding = torch.nn.ReflectionPad2d((0, pad_size[0],
                                0, pad_size[1]))
        
        self.shapes = [1, 4, 8, 16, 32, 64]

        levels = []

        for i in range(self.dsl):

            levels.append(self.create_subnetwork(ds_level=i, extra_conv=(i > 1)))

        
        self.resolution_levels = nn.ModuleList(levels)


    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.

        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).

        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.

        """
        
        c = self.fbp_layer(c)
        c = self.img_padding(c)
        
        outputs = []
        for m in self.resolution_levels:
            #print(m(c).shape)
            outputs.append(m(c))
        return outputs


    def create_subnetwork(self, ds_level, extra_conv=True, batchnorm=False):
        padding = [4,2,2,2,1,1,1]
        kernel = [9,5,5,5,3,3]

        modules = []
        
        for i in range(ds_level+1):
            #modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
            #print('ds_level: ', ds_level, ' in channel -> ', self.shapes[i], ' out channel -> ', self.shapes[i+1])
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=kernel[i], 
                                    padding=padding[i], 
                                    stride=2))
        
            #modules.append(nn.BatchNorm2d(self.shapes[i+1]))         
            modules.append(nn.LeakyReLU())
        #modules.append(InvertibleDownsampling2D(in_channels=1,stride=2, method='cayley', init='haar',
        #                                        learnable=True))
        #print('ds_level: ', ds_level, " output: ", self.shapes[ds_level+1])
        #if extra_conv: 
        #    modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1], 
        #                            out_channels=self.shapes[ds_level+1]*2, 
        #                            kernel_size=3, 
        #                            padding=1))
        #    modules.append(nn.ELU())
        #    modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1]*2, 
        #                            out_channels=self.shapes[ds_level+1], 
        #                            kernel_size=3, 
        #                            padding=1))
        #    modules.append(nn.ELU())
        #
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))
        #modules.append(nn.BatchNorm2d(self.shapes[ds_level+1]))
        return nn.Sequential(*modules)

class CondNet(nn.Module):
    """
    Conditional network H that sits on top of the invertible architecture.
    
    Attributes
    ----------
    resolution_levels : torch module list
        Building blocks of the conditional network.

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """
    
    def __init__(self, ray_trafo, img_size, downsample_levels=5):
        """
        

        Parameters
        ----------

        img_size : TYPE
            DESCRIPTION.


        Returns
        -------
        None.

        """
        super().__init__()
        
        self.img_size = img_size
        self.dsl = downsample_levels

        pad_size = (img_size[0] - ray_trafo.domain.shape[0],
                    img_size[1] - ray_trafo.domain.shape[1]) 
        self.img_padding = torch.nn.ReflectionPad2d((0, pad_size[0],
                                0, pad_size[1]))
        
        self.shapes = [1, 4, 8, 16, 32, 64]

        levels = []

        for i in range(self.dsl):

            levels.append(self.create_subnetwork(ds_level=i, extra_conv=(i > 1)))

        
        self.resolution_levels = nn.ModuleList(levels)


    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.

        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).

        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.

        """

        c = self.img_padding(c)
        
        outputs = []
        for m in self.resolution_levels:
            #print(m(c).shape)
            outputs.append(m(c))
        return outputs


    def create_subnetwork(self, ds_level, extra_conv=True, batchnorm=False):
        padding = [4,2,2,2,1,1,1]
        kernel = [9,5,5,5,3,3]

        modules = []
        
        for i in range(ds_level+1):
            #modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
            #print('ds_level: ', ds_level, ' in channel -> ', self.shapes[i], ' out channel -> ', self.shapes[i+1])
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=kernel[i], 
                                    padding=padding[i], 
                                    stride=2))
        
            #modules.append(nn.BatchNorm2d(self.shapes[i+1]))         
            modules.append(nn.LeakyReLU())
        #modules.append(InvertibleDownsampling2D(in_channels=1,stride=2, method='cayley', init='haar',
        #                                        learnable=True))
        #print('ds_level: ', ds_level, " output: ", self.shapes[ds_level+1])
        #if extra_conv: 
        #    modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1], 
        #                            out_channels=self.shapes[ds_level+1]*2, 
        #                            kernel_size=3, 
        #                            padding=1))
        #    modules.append(nn.ELU())
        #    modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1]*2, 
        #                            out_channels=self.shapes[ds_level+1], 
        #                            kernel_size=3, 
        #                            padding=1))
        #    modules.append(nn.ELU())
        #
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))
        #modules.append(nn.BatchNorm2d(self.shapes[ds_level+1]))
        return nn.Sequential(*modules)

            

    

def _add_conditioned_section(nodes, depth, in_ch, cond_level, conditions, coupling,
                             clamping=1.5):
    """
    Add conditioned notes to the network.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    depth : int
        Number of layer units in this block.
    in_ch : int
        Number of input channels.
    cond_level : int
        Current conditioning level.
    conditions : TYPE
        List of FrEIA condition notes.

    Returns
    -------
    None.

    """
    #print("----------------------------------")
    #print("Conditioned Section ", cond_level)
    #print("Input Channels: ", in_ch)
    #print("Hidden Channels in Coupling Block: ", in_ch*2)
    for k in range(depth):
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="ActNorm_{}_{}".format(cond_level, k)))


        nodes.append(Ff.Node(nodes[-1].out0, NICECouplingBlock,
                            {'F_args':{'leaky_slope': 5e-2, 'channels_hidden':in_ch*2, 'kernel_size': 3 if k % 2 == 0 else 1}},
                            conditions = conditions[cond_level],
                            name="NICEBlock_{}_{}".format(cond_level, k)))

        #nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv, 
        #                         {'M':random_orthog(in_ch)}, 
        #                         name='1x1Conv_{}_{}'.format(cond_level, k)))
        nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed':k}, 
                                 name='PermuteRandom{}_{}'.format(cond_level, k)))
        
def _add_downsample(nodes, downsample, in_ch, coupling, clamping=1.5):
    """
    Downsampling operations.

    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsample : str
        Downsampling method. Currently there are three options: 'haar', 
        'reshape' and 'invertible'.
    in_ch : int
        Number of input channels.
    clamping : float, optional
        The default value is 1.5.

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
    
    #print("-----------------")
    #print("Downsampling Section")
    #print("Coupling Block in shape: ", 4*in_ch)
    #print("Coupling Block hidden shape:, ", in_ch)
    for i in range(2):

        nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="DS_ActNorm_{}".format(i)))

        nodes.append(Ff.Node(nodes[-1].out0, NICECouplingBlock,
                            {'F_args':{'leaky_slope': 5e-2, 'channels_hidden':in_ch*2, 'kernel_size': 1}}, name="DS_NICECoupling_{}".format(i)))

        nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, {'seed':i}, 
                                 name='DS_PermuteRandom_{}'.format(i)))
    #
    #    nodes.append(Ff.Node(nodes[-1].out0, Fm.Fixed1x1Conv, 
    #                            {'M':random_orthog(4*in_ch)}, 
    #                            name='1x1Conv_downsample_{}'.format(i)))
        

