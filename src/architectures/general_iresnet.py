import sys

sys.path.append("../3rdparty/invertible-resnet")

import interfaces.torch_utils as utils

import numpy as np
import torch
import torch.nn as nn

from models.model_utils import injective_pad, ActNorm2D, Split
from models.model_utils import MaxMinGroup
from spectral_norm_conv_inplace import spectral_norm_conv
from spectral_norm_fc import spectral_norm_fc
from matrix_utils import exact_matrix_logarithm_trace, power_series_matrix_logarithm_trace
import pdb

def downsample_shape(shape, stride=2):
    return shape[:-3] + (shape[-3] * stride**2,
                         shape[-2] // stride,
                         shape[-2] // stride)

class conv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
        """
        Invertible bottleneck, adapted for multiple strides from
        https://github.com/jhjacobsen/invertible-resnet/blob/master/models/conv_iResNet.py

        Args:
            in_shape: shape of the input (channels, height, width)
            int_ch: dimension of intermediate layers
            stride: Stride for downsampling, positive integer. 1 for no downsampling.
            coeff: desired lipschitz constant
            input_nonlin: if true applies a nonlinearity on the input
            actnorm: if true uses actnorm like GLOW
            n_power_iter: number of iterations for spectral normalization
            nonlin: the nonlinearity to use
        """
        super().__init__()

        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape

        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 3 # kernel size for first conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size2 = 1 # kernel size for second conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0),
                                                  (int_ch, h, w), kernel_size2))
        layers.append(nonlin())
        kernel_size3 = 3 # kernel size for third conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride != 1:
            x = self.squeeze.forward(x)

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, trace + an_logdet

    def inverse(self, y, max_iter=100):
        """
        Returns the inverse of a given element.

        Args:
            max_iter: Maximum number of iterations for fixed-point inverse.
        """
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        for iter_index in range(max_iter):
            summand = self.bottleneck_block(x)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        if self.stride != 1:
            x = self.squeeze.inverse(x)
        return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)

class Squeeze(nn.Module):
    """
    Increases filter count by reducing size.
    """
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def inverse(self, input):
        return utils.reduce_filters(input, self.block_size)

    def forward(self, input):
        return utils.increase_filters(input, self.block_size)

    def squeeze_shape(self, shape):
        """
        Returns the expected output shape given a certain input shape.
        """
        return downsample_shape(shape, self.block_size)

class IResNet(nn.Module):
    """
    Changed version of the iResNet implementation in
    https://github.com/jhjacobsen/invertible-resnet/blob/master/models/conv_iResNet.py
    """
    def __init__(self, in_shape, n_blocks, strides, n_channels, init_ds=2, inj_pad=0,
                 coeff=.9,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 actnorm=True,
                 nonlin="relu"):
        """
        Args:
          init_ds: Initial down squeeze.
        """
        super().__init__()

        if not len(n_blocks) == len(strides) == len(n_channels):
            raise ValueError("n_blocks, strides and n_channels must have the length.")
        if in_shape[-1] % (init_ds * np.prod(strides)) != 0:
            raise ValueError("input width must be a multiple of the product of strides "
                "including init_ds")
        if in_shape[-2] % (init_ds * np.prod(strides)) != 0:
            raise ValueError("input height must be multiple of the product of strides "
                "including init_ds")

        self.init_ds = init_ds
        self.ipad = inj_pad
        self.n_blocks = n_blocks

        # parameters for trace estimation
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter

        self.init_squeeze = Squeeze(self.init_ds)
        self.inj_pad = injective_pad(inj_pad)
        if self.init_ds != 1:
           in_shape = downsample_shape(in_shape, self.init_ds)

        in_shape = (in_shape[0] + inj_pad, in_shape[1], in_shape[2])  # adjust channels

        self.stack, self.in_shapes, self.final_shape = \
            self._make_stack(n_channels, n_blocks, strides,
                             in_shape, coeff, conv_iresnet_block,
                             actnorm, n_power_iter, nonlin)

        # total_ds = init_ds * prod(strides)
        # final_shape = ((in_shape[0]+inj_pad) * total_ds**2,
        #                in_shape[1] // total_ds,
        #                in_shape[2] // total_ds)

    def _make_stack(self, n_channels, n_blocks, strides, in_shape, coeff, block,
                    actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        in_shapes = []
        for i, (int_dim, stride, blocks) in enumerate(zip(n_channels, strides, n_blocks)):
            for j in range(blocks):
                in_shapes.append(in_shape)
                block_list.append(block(in_shape, int_dim,
                                        numTraceSamples=self.numTraceSamples,
                                        numSeriesTerms=self.numSeriesTerms,
                                        # use stride if first layer in block else 1
                                        stride=(stride if j == 0 else 1),
                                        # add nonlinearity to input for all but first layer
                                        input_nonlin=(i + j > 0),
                                        coeff=coeff,
                                        actnorm=actnorm,
                                        n_power_iter=n_power_iter,
                                        nonlin=nonlin))
                if stride != 1 and j == 0:
                    in_shape = downsample_shape(in_shape, stride)

        return block_list, in_shapes, in_shape

    def get_in_shapes(self):
        return self.in_shapes

    def get_final_shape(self):
        return utils.flatten_shape(self.final_shape, batch=False)

    def forward(self, x, ignore_logdet=False):
        """
        IResNet forward pass.

        Returns: result, trace
        """
        if self.init_ds != 1:
            x = self.init_squeeze.forward(x)

        if self.ipad != 0:
            x = self.inj_pad.forward(x)

        z = x
        trace_sum = None
        for block in self.stack:
            with utils.grad_or_no_grad(None if ignore_logdet else True):
                z, trace = block(z, ignore_logdet=ignore_logdet)
            if trace_sum is None:
                trace_sum = trace
            else:
                trace_sum += trace

        if not ignore_logdet and not torch.is_grad_enabled():
            z = z.detach()

        return z.flatten(start_dim=1), trace_sum

    def inverse(self, z, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            x = z.view(z.shape[:1] + self.final_shape)
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, max_iter=max_iter)

            if self.ipad != 0:
                x = self.inj_pad.inverse(x)

            if self.init_ds != 1:
                x = self.init_squeeze.inverse(x)
        return x

    def set_num_terms(self, n_terms):
        for block in self.stack:
            for layer in block.stack:
                layer.numSeriesTerms = n_terms
