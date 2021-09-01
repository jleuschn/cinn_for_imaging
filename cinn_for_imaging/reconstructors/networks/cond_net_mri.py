from numpy.lib.arraypad import pad
import torch 
import torch.nn as nn 

from cinn_for_imaging.reconstructors.networks.layers import Flatten
from dival.reconstructors.networks.unet import UNet


class ResNetCondNet(nn.Module):
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
    
    def __init__(self, img_size, downsample_levels=5,cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):
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

        
        self.img_size = img_size
        self.dsl = downsample_levels
        

        
        
        # FBP and resizing layers
        self.unet_out_shape = 16

        self.img_size = img_size
        self.dsl = downsample_levels
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + cond_conv_channels
        self.use_fc_block = use_fc_block
        levels = []
        for i in range(self.dsl):   
            levels.append(self.create_subnetwork(ds_level=i))
        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl))

        self.preprocessing_net = nn.Sequential(
                ResBlock(in_ch=1, out_ch=8),
                ResBlock(in_ch=8, out_ch=8), 
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1),
                ResBlock(in_ch=8, out_ch=16),
                ResBlock(in_ch=16, out_ch=16),
                nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),                            
                ResBlock(in_ch=16, out_ch=self.unet_out_shape),
  
        )
        #UNet(in_ch=1, out_ch=self.net_out_shape, channels=[8, 16,16], skip_channels=[8,16,16], use_sigmoid=False, use_norm=True)
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
            
        
        outputs = []

        c_unet = self.preprocessing_net(c)

        for m in self.resolution_levels:
            #print(m(c_unet).shape)
            outputs.append(m(c_unet))
        return outputs


    def create_subnetwork(self, ds_level):

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=2))
        
            modules.append(ResBlock(in_ch=self.shapes[i+1], out_ch=self.shapes[i+1]))
            #modules.append(nn.BatchNorm2d(self.shapes[i+1]))         
            #modules.append(nn.LeakyReLU())

            #modules.append(nn.Conv2d(in_channels=self.shapes[i+1], 
            #                out_channels=self.shapes[i+1], 
            #                kernel_size=3, 
            #                padding=1, 
            #                stride=1))
            #modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))
        return nn.Sequential(*modules)



    def create_subnetwork_fc(self, ds_level):

        modules = []
        
        for i in range(ds_level):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=2))
            
            modules.append(ResBlock(in_ch=self.shapes[i+1], out_ch=self.shapes[i+1]))

            #modules.append(nn.BatchNorm2d(self.shapes[i+1]))         
            #modules.append(nn.LeakyReLU())

            #modules.append(nn.Conv2d(in_channels=self.shapes[i+1], 
            #                out_channels=self.shapes[i+1], 
            #                kernel_size=3, 
            #                padding=1, 
            #                stride=1))
            #modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(5,5))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)





class SimpleCondNetFBP(nn.Module):
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
    
    def __init__(self, img_size, downsample_levels=5, cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):
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
        
        self.img_size = img_size
        self.dsl = downsample_levels
        

        
        # FBP and resizing layers
        self.unet_out_shape = 16

        self.img_size = img_size
        self.dsl = downsample_levels
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + cond_conv_channels
        self.use_fc_block = use_fc_block
        levels = []
        for i in range(self.dsl):   
            levels.append(self.create_subnetwork(ds_level=i))
        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl))

        self.preprocessing_net = nn.Sequential(
                ResBlock(in_ch=1, out_ch=8),
                ResBlock(in_ch=8, out_ch=8), 
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1),
                ResBlock(in_ch=8, out_ch=16),
                ResBlock(in_ch=16, out_ch=16),
                nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),                            
                ResBlock(in_ch=16, out_ch=self.unet_out_shape),
  
        )
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
            
        
        outputs = []

        c = self.preprocessing_net(c)
        for m in self.resolution_levels:
            #print(m(c).shape)
            outputs.append(m(c))
        return outputs


    def create_subnetwork(self, ds_level):

        modules = []
    
        modules.append(nn.Conv2d(in_channels=self.unet_out_shape, 
                                out_channels=self.shapes[ds_level+1], 
                                kernel_size=3, 
                                padding=1, 
                                stride=1))

        for i in range(ds_level+1):
            modules.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))


        return nn.Sequential(*modules)



    def create_subnetwork_fc(self, ds_level):

        modules = []
        
        modules.append(nn.Conv2d(in_channels=self.unet_out_shape, 
                                    out_channels=self.shapes[ds_level+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=1))


        for i in range(ds_level):
            modules.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(5,5))
        modules.append(Flatten())

        return nn.Sequential(*modules)




class AvgPoolCondNetFBP(nn.Module):
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
    
    def __init__(self, img_size, downsample_levels=5, cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):
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
        
        self.img_size = img_size
        self.dsl = downsample_levels
        
        
        self.img_size = img_size
        self.dsl = downsample_levels
        self.fc_cond_dim = cond_fc_size
        self.shapes = [1] + cond_conv_channels
        self.use_fc_block = use_fc_block
        levels = []
        for i in range(self.dsl):   
            levels.append(self.create_subnetwork(ds_level=i))
        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl))

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
            
        
        outputs = []

        for m in self.resolution_levels:
            #print(m(c).shape)
            outputs.append(m(c))
        return outputs


    def create_subnetwork(self, ds_level):

        modules = []
        
        modules.append(nn.Conv2d(in_channels=1, 
                                    out_channels=self.shapes[ds_level+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=1))


        for i in range(ds_level+1):
            modules.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))


        return nn.Sequential(*modules)



    def create_subnetwork_fc(self, ds_level):

        modules = []
        
        modules.append(nn.Conv2d(in_channels=1, 
                                    out_channels=self.shapes[ds_level+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=1))


        for i in range(ds_level):
            modules.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(5,5))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)





class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, 2*out_ch, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*out_ch, 2*out_ch, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*out_ch, out_ch, 1, padding=0, stride=1))

        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1)
            
        self.final_activation = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        conv = self.conv_block(x)

        res = self.residual(x)

        y = self.batch_norm(conv + res)
        y = self.final_activation(y)
        
        return y
