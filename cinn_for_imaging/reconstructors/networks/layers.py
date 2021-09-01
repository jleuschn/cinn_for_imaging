"""
Includes important types of layers:
    - InvertibleDownsampling
    - NICECouplingBlock
    - F_Conv (used for NICECouplingBlock)
    - FBPModule
    - Flatten
"""

import torch
from torch.autograd import grad
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from scipy.stats import special_ortho_group

import warnings
from typing import Sequence, Union

from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule, TorchRayTrafoParallel2DModule
from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule

from cinn_for_imaging.util.iunets.layers import InvertibleDownsampling2D, InvertibleUpsampling2D

import FrEIA.modules as Fm

class InvertibleDownsampling(Fm.InvertibleModule):
    def __init__(self, dims_in, stride=2, method='cayley', init='haar',
                 learnable=True, *args, **kwargs):
        super().__init__(dims_in)
        self.stride = tuple(_pair(stride))
        self.invertible_downsampling = InvertibleDownsampling2D(in_channels=dims_in[0][0],
                                                                stride=stride,
                                                                method=method,
                                                                init=init,
                                                                learnable=learnable,
                                                                *args,
                                                                **kwargs)

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        log_jac_det = 0.
        if not rev:
            x = self.invertible_downsampling.forward(x)
        else:
            x = self.invertible_downsampling.inverse(x)

        return (x,), log_jac_det

    
    def output_dims(self, input_dims):
        """
        Calculates the output dimension of the invertible downsampling.
        Currently, only a stride of 2 is supported

        Parameters
        ----------
        input_dims : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*self.stride[0] * self.stride[1], w//self.stride[0], h//self.stride[1]
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class InvertibleUpsampling(Fm.InvertibleModule):
    def __init__(self, dims_in, stride=2, method='cayley', init='haar',
                 learnable=True, *args, **kwargs):
        super().__init__(dims_in)
        self.stride = tuple(_pair(stride))
        self.invertible_upsampling = InvertibleUpsampling2D(in_channels=dims_in[0][0],
                                                                stride=stride,
                                                                method=method,
                                                                init=init,
                                                                learnable=learnable,
                                                                *args,
                                                                **kwargs)

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        log_jac_det = 0.
        if not rev:
            x = self.invertible_upsampling.forward(x)
        else:
            x = self.invertible_upsampling.inverse(x)

        return (x,), log_jac_det

    
    def output_dims(self, input_dims):
        """
        Calculates the output dimension of the invertible downsampling.
        Currently, only a stride of 2 is supported

        Parameters
        ----------
        input_dims : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//(self.stride[0] * self.stride[1]), w*self.stride[0], h*self.stride[1]
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]





class F_conv(nn.Module):
    '''Not itself reversible, just used below'''

    def __init__(self, in_channels, out_channels, channels_hidden=None, kernel_size=3, leaky_slope=0.1):
        super().__init__()

        if not channels_hidden:
            channels_hidden = out_channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,
                               kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(channels_hidden, channels_hidden,
                               kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(channels_hidden, out_channels,
                               kernel_size=kernel_size, padding=pad)


    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv2(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv3(out)

        return out




"""
From: https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/coupling_layers.py
added argument c=[] to .jacobian()

and chanded assert in __init__ to same in GlowCouplingBlock. 
"""
class NICECouplingBlock(Fm.InvertibleModule):
    '''Coupling Block following the NICE design.
    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.'''

    def __init__(self, dims_in, dims_c=[],F_args={}):
        super().__init__(dims_in)

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.F = F_conv(self.split_len2 + condition_length, self.split_len1, **F_args)

        self.G = F_conv(self.split_len1 + condition_length, self.split_len2, **F_args)

    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))
        log_jac_det = 0.

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return [torch.cat((y1, y2), 1)], log_jac_det

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

    
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




class Fixed1x1ConvOrthogonal(Fm.InvertibleModule):
    '''Given an invertible matrix M, a 1x1 convolution is performed using M as
    the convolution kernel. Effectively, a matrix muplitplication along the
    channel dimension is performed in each pixel.
    
    The invertible matrix M is computed using scipy.stats.special_ortho_group and the shape is 
    automatically inferred from the data
    '''

    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]
        if self.channels > 512:
            warnings.warn(("scipy.stats.special_ortho_group  will take a very long time to initialize "
                           f"with {self.channels} feature channels."))

        M = special_ortho_group.rvs(self.channels)

        self.M = nn.Parameter(torch.FloatTensor(M).view(*M.shape, 1, 1), requires_grad=False)
        self.M_inv = nn.Parameter(torch.FloatTensor(M.T).view(*M.shape, 1, 1), requires_grad=False)
        #self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        #n_pixels = x[0][0, 0].numel()
        j = 0.0#self.logDetM * n_pixels
        # log abs det of special ortho matrix is always zero (det is either 1 or -1 )
        if not rev:
            return (F.conv2d(x[0], self.M),), j
        else:
            return (F.conv2d(x[0], self.M_inv),), -j

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError(f"{self.__class__.__name__} requires 3D input (channels, height, width)")
        return input_dims



class GradientDescentStep(Fm.InvertibleModule):
    '''
    An invertible step of gradient descent 
        x = z - step_size*A_adj(A z - y)
    with step_size < 1/|A|**2

    Only for CT
    '''

    def __init__(self, dims_in, op, step_size, dims_c=None):
        """
        op: ray_trafo (odl)
        """
        super().__init__(dims_in, dims_c)

        self.step_size = nn.Parameter(torch.tensor(step_size).float(), requires_grad=False)
        self.op = TorchRayTrafoParallel2DModule(op)
        self.op_adj = TorchRayTrafoParallel2DAdjointModule(op)

        self.fixed_point_iteration = 10

    def forward(self, x, c=[], rev=False, jac=True):
        j = 0.0 # gradient is constant w.r.t. parameters (is it really?)
        #if not rev:
        if rev:
            z = x[0]
            with torch.no_grad():
                for i in range(self.fixed_point_iteration - 1):
                    z = x[0] + self.step_size*(self.op_adj(self.op(z) - c[0]))

            return (x[0] + self.step_size*(self.op_adj(self.op(z) - c[0])),), j
        else:

            return (x[0] - self.step_size*self.op_adj(self.op(x[0]) - c[0]),), -j

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class Split(Fm.InvertibleModule):
    """Invertible split operation.

    Splits the incoming tensor along the given dimension, and returns a list of
    separate output tensors. The inverse is the corresponding merge operation.

    When section size is smaller than dims_in, a new section with the remaining dimensions is created. 
    """

    def __init__(self,
                 dims_in: Sequence[Sequence[int]],
                 section_sizes: Union[int, Sequence[int]] = None,
                 n_sections: int = 2,
                 dim: int = 0,
     ):
        """Inits the Split module with the attributes described above and
        checks that split sizes and dimensionality are compatible.

        Args:
          dims_in:
            A list of tuples containing the non-batch dimensionality of all
            incoming tensors. Handled automatically during compute graph setup.
            Split only takes one input tensor.
          section_sizes:
            If set, takes precedence over ``n_sections`` and behaves like the
            argument in torch.split(), except when a list of section sizes is given
            that doesn't add up to the size of ``dim``, an additional split section is
            created to take the slack. Defaults to None.
          n_sections:
            If ``section_sizes`` is None, the tensor is split into ``n_sections``
            parts of equal size or close to it. This mode behaves like
            ``numpy.array_split()``. Defaults to 2, i.e. splitting the data into two
            equal halves.
          dim:
            Index of the dimension along which to split, not counting the batch
            dimension. Defaults to 0, i.e. the channel dimension in structured data.
        """
        super().__init__(dims_in)

        # Size and dimensionality checks
        assert len(dims_in) == 1, "Split layer takes exactly one input tensor"
        assert len(dims_in[0]) >= dim, "Split dimension index out of range"
        self.dim = dim
        l_dim = dims_in[0][dim]

        if section_sizes is None:
            assert 2 <= n_sections, "'n_sections' must be a least 2"
            if l_dim % n_sections != 0:
                warnings.warn('Split will create sections of unequal size')
            self.split_size_or_sections = (
                [l_dim//n_sections + 1] * (l_dim%n_sections) +
                [l_dim//n_sections] * (n_sections - l_dim%n_sections))
        else:
            if isinstance(section_sizes, int):
                assert section_sizes < l_dim, "'section_sizes' too large"
            else:
                assert isinstance(section_sizes, (list, tuple)), \
                    "'section_sizes' must be either int or list/tuple of int"
                assert sum(section_sizes) <= l_dim, "'section_sizes' too large"
                if sum(section_sizes) < l_dim:
                    warnings.warn("'section_sizes' too small, adding additional section")
                    section_sizes.append(l_dim - sum(section_sizes))
            self.split_size_or_sections = section_sizes

    def forward(self, x, rev=False, jac=True):
        """See super class InvertibleModule.
        Jacobian log-det of splitting is always zero."""
        if rev:
            return [torch.cat(x, dim=self.dim+1)], 0
        else:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1), 0

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) == 1, "Split layer takes exactly one input tensor"
        # Assemble dims of all resulting outputs
        return [tuple(input_dims[0][j] if (j != self.dim) else section_size
                      for j in range(len(input_dims[0])))
                for section_size in self.split_size_or_sections]


if __name__ == "__main__":
    """
    SoftPermute = Fixed1x1ConvOrthogonal([[64, 256, 256]])

    x = torch.randn(1, 64, 256, 256)

    y, _ = SoftPermute([x], rev=False)

    x_re, _ = SoftPermute(y, rev=True)
    import numpy as np 
    print(np.allclose(x.numpy(), x_re[0].numpy(), atol=1e-06))
    print(np.linalg.norm((x.numpy() - x_re[0].numpy())**2))
    """

    import odl 
    from odl.contrib.torch import OperatorModule
    import numpy as np 
    import matplotlib.pyplot as plt 
    reco_space = odl.uniform_discr(
            min_pt=[-256, -256], max_pt=[256, 256], shape=[512, 512])

    phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    
    
    geometry = odl.tomo.parallel_beam_geometry(reco_space, 30)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    y = ray_trafo(phantom) 

    y = torch.from_numpy(np.asarray(y))
    y = y.unsqueeze(0).unsqueeze(0)
    x = torch.from_numpy(np.asarray(phantom))
    x = x.unsqueeze(0).unsqueeze(0)
    op = OperatorModule(ray_trafo)
    op_adj = OperatorModule(ray_trafo.adjoint)

    step_size = torch.tensor(1/(odl.power_method_opnorm(ray_trafo)**2*2)).float()

    gradient_descent_step = GradientDescentStep([1, 512, 512], op, op_adj, step_size)

    z = torch.zeros_like(x)
    print(x.shape, y.shape, z.shape)
    x1, _ = gradient_descent_step([z], c = [y], rev=True)

    print(x1[0].shape)

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(x[0,0,:,:].numpy(), cmap="gray")
    ax1.set_title("gt")
    ax2.imshow(x1[0][0,0,:,:].numpy(), cmap="gray")
    ax2.set_title("one step of gradient descent")

    plt.show()

    z, _ = gradient_descent_step([x], c = [y])
    x_re, _ = gradient_descent_step(z, c = [y], rev=True)


    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(x[0,0,:,:].numpy(), cmap="gray")
    ax1.set_title("x")
    ax2.imshow(x_re[0][0,0,:,:].numpy(), cmap="gray")
    ax2.set_title("xhat = F^-1(F(x))")
    plt.show()

    print(np.allclose(x.numpy(), x_re[0].numpy(), atol=1e-06))
    print(np.linalg.norm((x.numpy() - x_re[0].numpy())**2))