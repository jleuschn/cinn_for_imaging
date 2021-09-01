from numpy.lib.arraypad import pad
import torch 
import torch.nn as nn 

from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule
from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule

from cinn_for_imaging.reconstructors.networks.layers import Flatten
from dival.reconstructors.networks.unet import UNet


    
class FBPModule(torch.nn.Module):
    """
    Torch module of the filtered back-projection (FBP).

    Methods
    -------
    forward(x)
        Compute the FBP.
        
    """
    
    def __init__(self, ray_trafo, filter_type='Hann',
                 frequency_scaling=1.):
        super().__init__()
        self.ray_trafo = ray_trafo
        filter_op = fbp_filter_op(self.ray_trafo,
                          filter_type=filter_type,
                          frequency_scaling=frequency_scaling)
        self.filter_mod = OperatorModule(filter_op)
        self.ray_trafo_adjoint_mod = (
            TorchRayTrafoParallel2DAdjointModule(self.ray_trafo))
        
    def forward(self, x):
        x = self.filter_mod(x)
        x = self.ray_trafo_adjoint_mod(x)
        return x




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
    
    def __init__(self, ray_trafo, img_size, downsample_levels=5,
                 filter_type='Hann', frequency_scaling=1., cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):
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
        #self.preprocessing_net = UNet(in_ch=1, out_ch=self.unet_out_shape, channels=[8, 16,16], skip_channels=[8,16,16], use_sigmoid=False, use_norm=True)
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

        c = self.fbp_layer(c)
        c = self.img_padding(c)        
        c_unet = self.preprocessing_net(c)

        for m in self.resolution_levels:
            #print(m(c).shape)
            outputs.append(m(c_unet))
        return outputs


    def create_subnetwork(self, ds_level):
        padding = [2,2,2,2,1,1,1]
        kernel = [5,5,5,5,3,3]

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=kernel[i], 
                                    padding=padding[i], 
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
        padding = [2,2,2,2,1,1,1]
        kernel = [5,5,5,5,3,3]

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=kernel[i], 
                                    padding=padding[i], 
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
        modules.append(nn.AvgPool2d(6,6))
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
    
    def __init__(self, ray_trafo, img_size, downsample_levels=5,
                 filter_type='Hann', frequency_scaling=1., cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):
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

        c = self.fbp_layer(c)
        c = self.img_padding(c)        
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


        for i in range(ds_level+1):
            modules.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(6,6))
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
    
    def __init__(self, ray_trafo, img_size, downsample_levels=5,
                 filter_type='Hann', frequency_scaling=1., cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):
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

        c = self.fbp_layer(c)
        c = self.img_padding(c)        

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


        for i in range(ds_level+1):
            modules.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(6,6))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)
    

class CondUNet(nn.Module):
    """
    Conditional UNet architecture. Can be used for CT, MRI or any other
    application. Provide the classical reconstruction methods in the
    "prep_layers" argument
    
    """

    def __init__(self, prep_layers, downsample_levels:int,
                 use_sigmoid:bool = True):
        """
        

        Parameters
        ----------
        prep_layers : nn.Module
            Preparation layers, i.e. classical reconstruction method +
            reshaping to the correct image size.
        downsample_levels : int
            DESCRIPTION.
        use_sigmoid : bool, optional
            Use a sigmoid activation for the final output. The default is True.

        Returns
        -------
        None.

        """

        super().__init__()

        self.downsample_levels = downsample_levels
        self.use_sigmoid = use_sigmoid

        # Layers related to the classical inversion, e.g. FBP + Resizing
        self.prep_layers = prep_layers

        # Downsampling lane of the condNet
        self.downsampling = list()
        for i in range(self.downsample_levels - 1):
            self.downsampling.append(nn.Sequential(
                nn.Conv2d(2**i, 2**(i+1), 3, padding=1, stride=2),
                nn.LeakyReLU(),
                ResBlock(in_ch=2**(i+1), out_ch=2**(i+2)),
                ResBlock(in_ch=2**(i+2), out_ch=2**(i+3)),
                ResBlock(in_ch=2**(i+3), out_ch=2**(i+2)),
                ResBlock(in_ch=2**(i+2), out_ch=2**(i+2)),
                ))
        self.downsampling = nn.ModuleList(self.downsampling)

        # Upsampling lane
        self.upsampling = list()
        for i in reversed(range(1, self.downsample_levels - 1)):
            self.upsampling.append(nn.ModuleList([
                torch.nn.Upsample(size=None, scale_factor=2,
                                  mode='bilinear', align_corners=True),
                nn.Conv2d(2**(i+2), 2**(i), 1, padding=0, stride=1),
                nn.Sequential(ResBlock(in_ch=2**(i+1), out_ch=2**(i+2)),
                              ResBlock(in_ch=2**(i+2), out_ch=2**(i+3)),
                              ResBlock(in_ch=2**(i+3), out_ch=2**(i+2)),
                              ResBlock(in_ch=2**(i+2), out_ch=2**(i+1)))
                ]))
            
        self.upsampling.append(nn.ModuleList([
                torch.nn.Upsample(size=None, scale_factor=2,
                                  mode='bilinear', align_corners=True),
                nn.Conv2d(4, 1, 1, padding=0, stride=1),
                nn.Sequential(ResBlock(in_ch=1, out_ch=2),
                              ResBlock(in_ch=2, out_ch=2),
                              ResBlock(in_ch=2, out_ch=3),
                              ResBlock(in_ch=3, out_ch=3),
                              nn.Conv2d(3, 1, 3, padding=1, stride=1))
                ]))

        self.upsampling = nn.ModuleList(self.upsampling)

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

        # Prepare the input to the post-processing UNet
        x = self.prep_layers(c)

        outputs = list()
        splits = list()

        # Downsampling lane
        for i in range(self.downsample_levels - 2):
            x = self.downsampling[i](x)
            outputs.append(x)
            x, x_split = torch.split(x, int(x.shape[1]/2), dim=1)
            splits.append(x_split)

        # Bottom area
        x = self.downsampling[-1](x)
        outputs.append(x)

        # Upsampling lane
        for i in range(self.downsample_levels - 2):
            x = self.upsampling[i][0](x)  # Upsampling
            x = self.upsampling[i][1](x)  # Channel adaption conv
            x = torch.cat([x, splits[-i-1]], dim=1)  # Concat
            x = self.upsampling[i][2](x)  # ResBlock
            outputs.append(x)

        # Final reconstruction level
        x = self.upsampling[self.downsample_levels - 2][0](x)  # Upsampling
        x = self.upsampling[self.downsample_levels - 2][1](x)  # Channel adaption conv
        x = self.upsampling[self.downsample_levels - 2][2](x)  # ResBlocks
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        outputs.append(x)
        
        # Reverse the direction to map (Up_Cond, Down_Cond) -> (Down, Up) since
        # the direction of the conditional U-Net is reverse to the iUnet
        # Order: Downsampling -> descending spatial dim + upsamling ascending
        # spatial dim
        outputs = outputs[::-1]

        return outputs


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size=3):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, 2*out_ch, size, padding=int(size/2), stride=1),
            nn.BatchNorm2d(2*out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(2*out_ch, 2*out_ch, size, padding=int(size/2), stride=1),
            nn.BatchNorm2d(2*out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(2*out_ch, out_ch, 1, padding=0, stride=1))

        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1)
            
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.final_activation = nn.LeakyReLU()

    def forward(self, x):
        conv = self.conv_block(x)

        res = self.residual(x)

        y = self.batch_norm(conv + res)
        y = self.final_activation(y)
        
        return y
