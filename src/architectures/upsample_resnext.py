import torch

import numpy as np

from interfaces.simple_model import SimpleTorchModel
from interfaces.torch_utils import apply_layer, view_batch
from interfaces.math_utils import round_up_integer_divide

class ResizeConvolution(torch.nn.Module):
  def __init__(self, filters_in, filters_out, kernel_size, scale_factor,
    padding=(0,0),
    groups=1, bias=True):
    super().__init__()

    def is_identity(scale_factor):
      if scale_factor == 1.:
        return True

      if type(scale_factor) == tuple or type(scale_factor) == list:
        for s in scale_factor:
          if s != 1.:
            return False
        return True

      return False

    if is_identity(scale_factor):
      self.upsample = None
    else:
      self.upsample = torch.nn.Upsample(scale_factor=scale_factor)

    self.conv = torch.nn.ConvTranspose2d(
      filters_in,
      filters_out,
      kernel_size,
      padding=padding,
      groups=groups,
      bias=bias
    )

  def forward(self, x):
    x = apply_layer(self.upsample, x)
    x = apply_layer(self.conv, x)

    return x

class UpsampleConvBlock(torch.nn.Module):
  def __init__(self, filters_in, filters_bottleneck, filters_out,
      groups, mid_kernel_size=(3,3), scale_factor=1.,
      leaky_slope=0.01, bn=False):
    super().__init__()
    
    if filters_in != filters_out or scale_factor != 1.:
      self.resample = ResizeConvolution(
        filters_in,
        filters_out,
        (1,1),
        scale_factor,
        bias=False
      )
    else:
      self.resample = None
    
    if leaky_slope == 0.:
      self.relu = torch.nn.ReLU()
    else:
      self.relu = torch.nn.LeakyReLU(
        negative_slope=leaky_slope)
    
    self.conv1 = torch.nn.Conv2d(
      filters_in,
      filters_bottleneck,
      (1,1),
      bias=False
    )
    
    self.conv2 = ResizeConvolution(
      filters_bottleneck,
      filters_bottleneck,
      mid_kernel_size,
      scale_factor,
      padding=(
        (mid_kernel_size[0] - 1) // 2,
        (mid_kernel_size[1] - 1) // 2
      ),
      groups=groups,
      bias=False
    )
    
    self.conv3 = torch.nn.Conv2d(
      filters_bottleneck,
      filters_out,
      (1,1),
      bias=False
    )
    
    if bn:
      self.bn1 = torch.nn.BatchNorm2d(filters_in)
      self.bn2 = torch.nn.BatchNorm2d(filters_bottleneck)
      self.bn3 = torch.nn.BatchNorm2d(filters_bottleneck)
    else:
      self.bn1 = None
      self.bn2 = None
      self.bn3 = None

  def forward(self, x):
    identity = apply_layer(self.resample, x)
    
    # using pre-activation like in
    # https://arxiv.org/pdf/1603.05027.pdf
    x = apply_layer(self.bn1, x)
    x = apply_layer(self.relu, x)
    x = apply_layer(self.conv1, x)
    
    x = apply_layer(self.bn2, x)
    x = apply_layer(self.relu, x)
    x = apply_layer(self.conv2, x)
    
    x = apply_layer(self.bn3, x)
    x = apply_layer(self.relu, x)
    x = apply_layer(self.conv3, x)
    
    return x + identity

def deconv_block(
  depth,
  filters_in, filters_bottleneck, filters_out,
  filters_middle=None,
  width=1,
  scale_factor=2.,
  **kwargs):
  if filters_middle is None:
    filters_middle = filters_out
  
  blocks = []
  
  blocks.append(UpsampleConvBlock(
    filters_in,
    filters_bottleneck,
    filters_middle if depth > 1 else filters_out,
    groups=width,
    scale_factor=scale_factor,
    **kwargs
  ))
  
  for i in range(depth - 1):
    blocks.append(UpsampleConvBlock(
      filters_middle,
      filters_bottleneck,
      filters_out if i == depth-2 else filters_middle,
      groups=width,
      **kwargs
    ))
  
  return torch.nn.Sequential(*blocks)

class FullyConnected(torch.nn.Linear):
  """
  Fully connected layer that operates on non-flattened data.
  """
  def __init__(self, input_shape, out_shape, **kwargs):
    self.__out_shape = out_shape
    super().__init__(np.prod(input_shape), np.prod(out_shape))

  def forward(self, x):
    return view_batch(
      super().forward(x.flatten(start_dim=1)),
      self.__out_shape)

class UpsampleResNeXt(torch.nn.Module):
  """
  An upsampling residual network based on ResNeXt.
  """
  def __init__(self, input_shape, output_shape,
               block_depths=(2,2,2,2), groups=4,
               width=4,
               bottleneck_factor=2,
               block_scale=2,
               block_filter_reduction=2,
               use_bn=False, leaky_slope=0.01
    ):
    """
    Initializes the UpsampleResNeXt.
    
    Args:
      input_shape: Expected input shape.
      output_shape: Desired shape of the output.
      groups: Number of groups used in residual units.
      width: Minimum bumber of filters per group.
      block_depths: Number of residual units in each block, from input to
        output layers, i.e. starting at the block after the fully connected
        layer.
      bottleneck_factor: Factor for reduction of filters in each bottleneck.
      block_scale: Scale factor of each residual block. May be a list or
        tuple in which case the length of the list must be equal to that of
        block_depths. If a single number is given each block uses the same
        scaling except for the first block which does not apply scaling.
        Both image dimensions will be scaled by the same factor.
      block_filter_reduction: Factor by which each block reduces the filters.
        May be a list or a single value, in the latter case each block except
        the first uses the same reduction and no reduction is done in the
        first block.
      use_bn: Whether to apply batch normalization before non-linearities.
      leaky_slope: Slope of the leaky ReLU non-linearities. 0 to use ReLU.  
    """
    super().__init__()
    
    def broadcast_to_blocks(param, name, base_types, defaults=[]):
      if type(param) in base_types:
        params = tuple(defaults) + (param,) * (len(block_depths) - len(defaults))
        # in case there were more defaults than base_types:
        # cut off unnecessary elements
        return params[:len(block_depths)]
      elif len(param) != len(block_depths):
        raise ValueError("Length of " + name +
                 " (" + str(len(param)) + ") must be equal to "
                "length of block_depths (" +
                 str(len(block_depths)) + ").")
      return param
    
    block_scale = broadcast_to_blocks(
      block_scale, "block_scale", [int, float], [1])
    block_filter_reduction = broadcast_to_blocks(
      block_filter_reduction, "block_filter_reduction", [int], [1])
    bottleneck_factor = broadcast_to_blocks(
      bottleneck_factor, "bottleneck_factor", [int, float])
    
    block_factor = int(np.round(np.prod(block_scale)))

    filters = int(width * groups * np.prod(block_filter_reduction) * np.max(bottleneck_factor))

    self.__start_shape = (filters,
                          # same factor for both output dimensions
                          output_shape[-2] // block_factor,
                          output_shape[-1] // block_factor)
    # shorthand for convenience
    start_shape = self.__start_shape

    if len(input_shape) == 3 and input_shape[-2] * input_shape[-1] > 1:
      # use down/upsampling to get input into right shape
      reshaping_layers = []

      curr_shape = input_shape

      if curr_shape[-2] > start_shape[-2] or curr_shape[-1] > start_shape[-1]:
        # downsample shape so that the current input fits into the intended one
        
        stride = (
          round_up_integer_divide(curr_shape[-2], start_shape[-2]),
          round_up_integer_divide(curr_shape[-1], start_shape[-1])
        )
        padding = stride[0] - 1, stride[1] - 1
        kernel_size = stride[0] * 2 - 1, stride[1] * 2 - 1

        new_shape = (curr_shape[-2] // stride[-2],curr_shape[-1] // stride[-1])

        if new_shape == start_shape[1:]:
          new_filters = curr_shape[0]
        else:
          # will be followed by upsampling and another convolution
          # this should lose as few information as possible
          new_filters = max(curr_shape[0], start_shape[0])

        reshaping_layers.append(torch.nn.Conv2d(
          curr_shape[0],
          new_filters,
          kernel_size,
          padding=padding,
          stride=stride,
          bias=True
        ))
        curr_shape = (new_filters,) + new_shape

        assert curr_shape[-2] <= start_shape[-2]
        assert curr_shape[-1] <= start_shape[-1]
      if curr_shape != start_shape:
        scale_factor = start_shape[-2] / curr_shape[-2], start_shape[-1] / curr_shape[-1]
        kernel_size = int(np.round(scale_factor[-2])) * 2 - 1, int(np.round(scale_factor[-1])) * 2 - 1
        padding = (kernel_size[-2] - 1) // 2, (kernel_size[-1] - 1) // 2

        # remaining operations can be done by an upsample
        reshaping_layers.append(ResizeConvolution(
          curr_shape[0],
          start_shape[0],
          kernel_size,
          scale_factor,
          padding=padding
        ))

      self.reshape = torch.nn.Sequential(*reshaping_layers)
    else:
      # fall back to fully connected layer
      self.reshape = FullyConnected(input_shape, self.__start_shape)
    
    blocks = []
    
    for d, scale, red, bttlf in zip(block_depths, block_scale,
                                    block_filter_reduction,
                                    bottleneck_factor):
      filters_out = filters // red
      
      blocks.append(deconv_block(
        d,
        filters,
        int(filters_out // bttlf),
        filters_out,
        width=groups,
        scale_factor=scale,
        bn=use_bn,
        leaky_slope=leaky_slope
      ))
      
      filters = filters_out
    
    self.blocks = torch.nn.Sequential(*blocks)
    
    self.conv = torch.nn.Conv2d(filters, 3, (1,1))

    self.lelu = torch.nn.LeakyReLU(negative_slope=leaky_slope)
    self.sig = torch.nn.Sigmoid()
  
  def forward(self, x):
    x = self.reshape(x)
    x = self.lelu(x)
    
    x = self.blocks(x)
    
    x = self.conv(x)
    x = self.sig(x)
    
    return x

class ReproducibleUpsampleResNeXt(SimpleTorchModel):
  def __init__(self, /, out_shape=None, name="Deconvolutional ResNeXt",
    parameters={}, **kwargs):
    """
    Create a new PictureCreationModel.
    """
    # merge existing parameters with defaults for parameters of the MiniResnet
    # default values will overwritten by existing values in parameters
    parameters = {
      **{
      "optimizer": "SGD",
      "weight_decay": 1e-4,
      "lr": 0.01,
      "epochs": 10,
      "loss": "L2",
      "_ReproducibleUpsampleResNeXt__version": "1.3.3",
      "output_shape": out_shape,
      "groups": 16,
      "width": 4,
      "block_depths": [3,4,4,3],
      "batchnorm": True,
      "leaky_slope": 0.01,
      "block_scale": 2,
      "block_reduction": 2,
      "bottleneck_factor": 2,
    },
    **parameters}

    super().__init__(parameters=parameters, name=name, **kwargs)

  def _initialize_model(self):
    self.model = UpsampleResNeXt(
      input_shape=self._get_parameter("input")[0][0].shape,
      output_shape=self._get_parameter("output_shape"),
      width=self._get_parameter("width"),
      groups=self._get_parameter("groups"),
      block_depths=self._get_parameter("block_depths"),
      use_bn = self._get_parameter("batchnorm"),
      leaky_slope = self._get_parameter("leaky_slope"),
      block_scale = self._get_parameter("block_scale"),
      block_filter_reduction = self._get_parameter("block_reduction"),
      bottleneck_factor=self._get_parameter("bottleneck_factor")
    )

    super()._initialize_model()
