import torch
import numpy as np

import random

import functools

class View(torch.nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(x.shape[0], *self.shape)

class grad_or_no_grad:
  def __init__(self, have_grad):
    self.have_grad = have_grad
    self.prev = torch.is_grad_enabled()

  def __enter__(self):
    if self.have_grad is not None:
      torch.set_grad_enabled(self.have_grad)

  def __exit__(self, *args):
    if self.have_grad is not None:
      torch.set_grad_enabled(self.prev)
    return False

class grad:
  def __init__(self):
    self.prev = torch.is_grad_enabled()

  def __enter__(self):
    torch.set_grad_enabled(True)

  def __exit__(self, *args):
    torch.set_grad_enabled(self.prev)
    return False

  def __call__(self, func):
    @functools.wraps(func)
    def decorate_with_grad(*args, **kwargs):
        with self:
            return func(*args, **kwargs)
    return decorate_with_grad

def on_correct_device(use_cuda, *args):
  """
  Puts the inputs onto cuda if gpu is desired and onto cpu otherwise.
  """
  return map(lambda x: x.cuda() if use_cuda else x.cpu(), args)

def to_batch(x):
  """
  Put a single tensor into a batch.
  """
  return x.view((1,) + x.shape)

def from_batch(x):
  """
  Convert from a 1-element batch to a single element tensor.
  """
  return x.view(x.shape[1:])

def apply_layer(layer, x):
    if layer is not None:
        return layer(x)
    else:
        return x

def view_batch(x, shape):
  """
  Preserves the batch dimension and vies every element in the given shape.
  """
  return x.view(x.shape[0], *shape)

def accumulate_mean(mean1, n1, mean2, n2):
  """
  Given two mean values and the number of elements the respective mean had been
  taken over, return a combined mean and the combined number of elements.
  """
  n_new = n1 + n2
  return mean1/n_new*n1 + mean2/n_new*n2, n_new

def agreements(outputs, labels):
  """
  Returns the number of agreements between the given model output with one-of
  class representations and integer labels for those classes.

  The outputs are taken to represent the first class with the highest number
  at its index in the representation.
  """
  return torch.count_nonzero(torch.argmax(outputs, dim=-1) == labels)

def accuracy(outputs, labels):
  """
  Returns the accuracy, i.e. the percentage of agreements between the labels
  and the outputs. Labels should be integers and outputs one-of representations
  of size fitting to the range of the class labels.
  """
  return agreements(outputs, labels) / len(labels)

def batch_count(input_size, batch_size):
  """
  Returns the total number of batches with the given number of inputs and the
  given batch size.
  """
  return (input_size - 1) // batch_size + 1

def get_batch_from_dataloader(dataloader, batch_size):
    """
    Receives a batch of the desired size from a dataloader.
    """
    sub_batches = []

    seen = 0

    for x, y in dataloader:
        in_this = max(0, min(batch_size - seen, len(x)))
        sub_batches.append(x[:in_this])

        seen += in_this
        if seen >= batch_size:
            break

    return torch.cat(sub_batches)

def reduce_filters(x, block_size=2):
  """
  Inverse operation to increase_filters. Puts together blocks from filters
  and thus reduces the amount of filters.
  """
  c, h, w = x.shape[-3:]
  bs = block_size
  sq_bs = bs*bs
  if c % sq_bs != 0:
    raise RuntimeError(
      "Block size '" + str(bs) + "' is incompatible with channel count '"
      + str(c) + "'. Channel count must be a multiple of squared block size.")
  constant_shape = x.shape[:-3]

  n_c = c // sq_bs
  output = x.view(constant_shape + (bs, bs, n_c, h, w))

  di = len(x.shape) - 3
  output = output.permute(*range(di), 2+di, 3+di, 0+di, 4+di, 1+di)

  return output.reshape(constant_shape + (n_c, h*bs, w*bs))


def increase_filters(x, block_size=2):
  """
  Reduces the last two dimensions of the tensor to 1/block_size by reordering
  them into the third to last dimension.

  Viewing the last three dimensions as channels, height, width, each
  blocks_size x blocks_size sized block will be split into channels in left to
  right, top to bottom order. Channel order will be all channels of the first
  slice, then all channels of the second etc.
  
    A B A B
    C D C D ->  stack(A A, B B, C C, D D)
    A B A B           A A  B B  C C  D D
    C D C D
  """
  # start with shape
  # ..., c, h, w
  c, h, w = x.shape[-3:]
  constant_shape = x.shape[:-3]
  bs = block_size
  if h % bs != 0:
    raise RuntimeError(
      "Block size '" + str(bs) + "' is incompatible with height '"
      + str(h) + "'. Height must be a multiple of block size.")
  if w % bs != 0:
    raise RuntimeError(
      "Block size '" + str(bs) + "' is incompatible with width '"
      + str(w) + "'. Width must be a multiple of block size.")

  # view as
  # ..., c, h/bs, bs, w/bs, bs
  output = x.view(constant_shape + (c, h//bs, bs, w//bs, bs))

  # permute to
  # ..., bs, bs, c, h/bs, w/bs
  di = len(x.shape) - 3
  output = output.permute(*range(di), 2+di, 4+di, 0+di, 1+di, 3+di)

  # view as
  # ..., c*bs*bs, h/bs, w/bs
  return output.reshape(constant_shape + (c*bs*bs, h//bs, w//bs))

def print_full_tensor(x, format_string=None):
  """
  Prints a full tensor onto the terminal, formatting according to its content.
  """
  # determine number of relevant digits
  # create format string
  def format_number(val, format_string, elem_width):
    return format_string.format(val, width=elem_width)

  def format_bool(val, format_string, elem_width):
    return format_string.format("■" if val else "□", width=elem_width)

  def format_complex(val, format_string, elem_width):
    real, imag = torch.view_as_real(val)
    ret = format_string.format(real, width=elem_width)
    if imag > 0:
      ret += " + " + format_string.format(imag, width=elem_width)
    else:
      ret += " - " + format_string.format(-imag, width=elem_width)
    return ret + "i"

  def format_int(val, format_string, elem_width):
    return format_string.format(val, width=elem_width).replace("_", "'")

  if format_string is None:
    default_format = lambda x: x
  else:
    default_format = lambda x: format_string

  def print_tensor_rec(x, format_function, format_string, elem_length,
                       indent=0, pre_string="", post_string=""):
    dim_count = len(x.shape)
    if dim_count == 0:
      # print the single element tensor directly
      format_function(x, format_string, elem_length)
    else:
      l = x.shape[0]
      line_start = " "*(indent-len(pre_string)) + pre_string

      if l == 0:
        # print empty tensor
        print(line_start + "[ ]" + post_string)
      elif dim_count == 1:
        # print a line
        l = x.shape[0]
        line_text = line_start + "[ "
        if l > 0:
          line_text += format_function(x[0], format_string, elem_length)
        for i in range(1,l):
          line_text += ", " + format_function(x[i], format_string, elem_length)
        line_text += "]" + post_string
        print(line_text)
      else:
        # print a block in square brackets using recursive calls
        indent += 1
        print_tensor_rec(x[0], format_function, format_string, elem_length,
                         indent, pre_string + "[",
                         "," if l > 1 else "]" + post_string)
        for i in range(1,l-1):
          for _ in range(dim_count - 2):
            print()
          print_tensor_rec(x[i], format_function, format_string, elem_length,
                           indent, "", ",")
        if l > 1:
          for _ in range(dim_count - 2):
            print()
          print_tensor_rec(x[-1], format_function, format_string, elem_length,
                           indent, "",
                           "]" + post_string)

  if x.dtype == torch.bool:
    f = format_bool
    f_str = default_format("{} ")
  elif (x.dtype == torch.float32 or x.dtype == torch.float64 or
      x.dtype == torch.float16 or x.dtype == torch.bfloat16):
    f = format_number
    f_str = default_format("{:{width}.5}")
  elif (x.dtype == torch.complex32 or x.dtype == torch.complex64 or
        x.dtype == torch.complex128):
    f = format_complex
    f_str = default_format("{:{width}.3}")
  elif (x.dtype == torch.int8 or x.dtype == torch.int16 or
        x.dtype == torch.int32 or x.dtype == torch.int64 or
        x.dtype == torch.uint8):
    # integer type
    # get max length
    f = format_int
    f_str = default_format("{:{width}_}")
  else:
    print("Tensor of unknown type {}".format(x.dtype))
    return

  x_real = torch.view_as_real(x) if x.is_complex() else x
  per_elem_f = format_number if x.is_complex() else f

  # maximum length of the string representation of a single element
  max_length = max(map(lambda elem: len(per_elem_f(elem, f_str, 0)),
                   x_real.flatten()))

  print_tensor_rec(x, f, f_str, max_length)

class BufferedSampler():
  """
  Gets samples from a sampling function in fixed size batches and buffers them
  to return arbitrary size batches.

  Args:
    sample_function: An object with a sample() function which can be called with
      the sample batch size as only parameter and returns a tensor of samples.
  """
  def __init__(self, sampler, batch_size):
    self.__sampler = sampler
    
    self.__buffer_size = batch_size
    self.__buffer_idx = self.__buffer_size

  def sample(self, n=None, *, device=None):
    """
    Samples either a single sample or a batch of samples of the specified size.

    Args:
      n: Number of samples to return as a batch. Returns a single sample in
        non-batched format if None.
      device: The desired device of the returned tensor.
    """
    if n is None:
      n = 1
      unbatch = True
    else:
      unbatch = False
      if n <= 0:
        return torch.empty(0)

    sample_count = 0

    while sample_count < n:
      if self.__buffer_idx == self.__buffer_size:
        with torch.no_grad():
          self.__buffer = self.__sampler.sample(self.__buffer_size)
        self.__buffer_idx = 0

      # number of samples to take from the currently buffered batch
      batch_n = min(
        # number of samples still needed
        n - sample_count,
        # number of samples still buffered
        self.__buffer_size - self.__buffer_idx)

      if sample_count == 0:
        samples = torch.empty((n,) + self.__buffer.shape[1:], device=device)

      samples[sample_count:sample_count + batch_n] = \
        self.__buffer[self.__buffer_idx:self.__buffer_idx + batch_n]

      sample_count += batch_n
      self.__buffer_idx += batch_n

    if unbatch:
      return samples[0]
    else:
      return samples

def flatten_shape(shape, batch=True):
  """
  Returns the flattened version of shape, i.e. viewed as a single array,
  one-dimensional array.

  Args:
    batch: Whether to interpret the first dimension as batch dimension or to
      include it in the flattening process.
  """
  if not batch:
    return (np.prod(shape),)
  else:
    return shape[:1] + (np.prod(shape[1:]))

def set_optimizer_param(opt, param_name, param_val):
  """
  Sets the specified parameter of an optimizer, e.g. 'lr', to the specified
  value for all parameter groups of the optimizer.
  """
  for g in opt.param_groups:
    g[param_name] = param_val

def get_optimizer_param(opt, param_name):
  """
  Return a list of value of the given parameter per parameter group.
  """
  results = []
  for g in opt.param_groups:
    results.append(g[param_name])
  return results

def upshape(tensor, target_shape):
  """
  The shape of the input tensor will be matched to the target shape by zero
  padding on a 1-dimensional view of the tensor (excluding batch dimension) and
  then reshaping the tensor to match the target shape.

  Will raise a ValueError if the size of the tensor is greater than the target
  size.

  Args:
    tensor: Input tensor to put into shape as batched input.
    target_shape: Shape that the resulting tensor should have without batch
      dimension.
  """
  input_shape = tensor.shape

  if target_shape == input_shape[1:]:
    return tensor

  total_target_size = np.prod(target_shape)
  # ignore batch dimension when calculating tensor size
  total_existing_size = np.prod(input_shape[1:])

  if total_target_size < total_existing_size:
    raise ValueError("Target shape " + str(target_shape) + " must not "
      "be smaller than input shape. Got shape " + str(input_shape[1:]))

  # will be >= 0 as total_target_size >= total_existing_size
  dsize = total_target_size - total_existing_size

  # dsize // 2 is dsize / 2 rounded down, (dsize+1) // 2 is that rounded up
  # sum of the two will always be dsize
  tensor = torch.nn.functional.pad(tensor.flatten(start_dim=1),
                                ((dsize+1) // 2, dsize // 2))

  return tensor.reshape(input_shape[:1] + target_shape)

def get_fixed_samples(dataset, indices):
  samples = []
  
  for i in indices:
    samples.append(dataset[i][0])

  return torch.stack(samples)

def get_samples(dataset, count):
  return get_fixed_samples(dataset, random.sample(range(len(dataset)), count))
