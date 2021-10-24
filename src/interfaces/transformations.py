import pickle

from functools import partial
from os.path import join as join_paths

import torch
import numpy as np

from .reproducible import Reproducible, VirtualReproducible
from . import torch_utils as utils
from .torch_utils import to_batch, from_batch
from .math_utils import round_up_integer_divide
from .command_line import progress_bar

class TupleUnpack(VirtualReproducible):
  def __init__(self, id, parameters={}, **kwargs):
    parameters = {
      **{
        "_TupleUnpack__version": "1.0.0",
        "id": id,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def __call__(self, x):
    return x[self._get_parameter("id")]

class ConcatOnSubset(VirtualReproducible):
  """
  Applies f1 on a subset of the arguments passed to f2.
  """
  def __init__(self, f1, f2, input_set, output_set=None, parameters={}, **kwargs):
    """
    Args:
      input_set: Ids of the inputs to be used in the order they should be passed
        to the function.
      output_set: Ids that the outputs should be using for the new inputs (outputs
        of the function) in the order of return values. Defaults to be the input
        set.
    """
    parameters = {
      **{
        "_ConcatOnSubset__version": "1.0.0",
        "input_set": input_set,
        "output_set": output_set if output_set is not None else input_set,
        "f1": f1,
        "f2": f2,
      },
      **parameters
    }

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__input_set = self._get_parameter("input_set")
    self.__output_set = self._get_parameter("output_set")
    self.__f1 = self._get_parameter("f1")
    self.__f2 = self._get_parameter("f2")

  def __call__(self, *args, **kwargs):
    self.ensure_available()
    tmp_args = []

    input_set = self.__input_set

    if type(input_set) == int:
      input_set = [input_set]

    for in_i in input_set:
      tmp_args.append(args[in_i])

    f1outs = self.__f1(*tmp_args, **kwargs)

    output_set = self.__output_set

    if type(output_set) == int:
      output_set = [output_set]

    if len(output_set) == 1:
      f1outs = [f1outs]

    new_args = []
    i_old = 0

    # ids of new values in the order they should be inserted
    new_ids = sorted(range(len(output_set)), key=output_set.__getitem__)

    def get_next_new(new_ids):
      if len(new_ids) == 0:
        return None, None, None

      next_new_id = new_ids[0]

      return output_set[next_new_id], f1outs[next_new_id], new_ids[1:]

    next_outid, next_new, new_ids = get_next_new(new_ids)

    for i_new in range(len(args) + len(output_set) - len(input_set)):
      if i_new == next_outid:
        new_args.append(next_new)
        next_outid, next_new, new_ids = get_next_new(new_ids)
      else:
        while i_old in input_set:
          i_old += 1
        new_args.append(args[i_old])
        i_old += 1

    return self.__f2(*new_args, **kwargs)

class Multiply(VirtualReproducible):
  """
  Right multiplication
  """
  def __init__(self, factor, parameters={}, **kwargs):
    parameters = {
      **{
        "_Multiply__version": "1.0.0",
        "factor": factor
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def __call__(self, val):
    return val * self._get_parameter("factor")

class ConfiguredTransformation(VirtualReproducible):
  """
  A transformation that is constructed by calling any method of an object and
  can poss additional fixed arguments to that method.
  """
  def __init__(self, parameters={}, **kwargs):
    """
    Get a new transformed data set.

    Args:
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "transformer": Object that performs the operation.
        - "call_method": If not None, the given method will be called instead
            of directly calling the transformation object.
        - "kwargs": Dictionary of additional arguments for the transformation
          function.
        - "make_batch": If set, the transformation function will receive data
          as single element batches instead.
    """
    parameters = {
      **{
        "_ConfiguredTransformation__version": "1.0.0",
        "transformer": None,
        "call_method": None,
        "kwargs": {},
        "make_batch": False
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    t = self._get_parameter("transformer")
    if self._get_parameter("call_method") is not None:
      t = getattr(t, self._get_parameter("call_method"))

    t = partial(t, **self._get_parameter("kwargs"))

    if self._get_parameter("make_batch"):
      t = partial(lambda f, x: from_batch(f(to_batch(x))), t)

    self.__transform = t

  def __call__(self, val):
    self.ensure_available()
    return self.__transform(val)

class TransformChain(VirtualReproducible):
  """
  Concatenation of any (fixed) number of transformations.
  """
  def __init__(self, *args, parameters={}, **kwargs):
    parameters = {
      **{
        "_TransformChain__version": "1.0.0"
      },
      **parameters}

    self.__param_count = 0

    for i, f in enumerate(args):
      parameters[i] = f
      self.__param_count += 1

    super().__init__(parameters=parameters, **kwargs)

  def __call__(self, val):
    for i in range(self.__param_count):
      val = self._get_parameter(i)(val)

    return val

class TransformConcat(VirtualReproducible):
  """
  Concatenation of two transformations.
  """
  def __init__(self, first=None, second=None, parameters={}, **kwargs):
    """
    Args:
      parameters:
        - "first": First transformation to perform.
        - "second": Second transformation to perform.
    """
    parameters = {
      **{
        "_TransformConcat__version": "1.0.0",
        "first": first,
        "second": second
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    pass

  def __call__(self, *args, **kwargs):
    self.ensure_available()
    return self._get_parameter("second")(self._get_parameter("first")(*args, **kwargs))

class MinMaxNormalization(Reproducible):
  """
  A transformation that scales and shifts a dataset to have the given min and
  max values per component.
  """
  def __init__(self, parameters={},
    batch_size=128,
    **kwargs):
    """
    Args:
      batch_size: Batch size when computing data min/max.
      parameters: Dictionary of parameters to set in the reproducible.
        Parameters for MinMaxNormalization are:
        - dataset: Dataset which should be stretched.
        - min: Minumum value each component of the data should have after
          stretching.
        - max: Maximum value each component of the data should have after
          stretching.
    """
    parameters = {**{
        "_MinMaxNormalization__version": "1.0.2",
        "dataset": None,
        "min": 0.,
        "max": 1.
      },
      **parameters}

    self.__batch_size = batch_size
    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    ds = self._get_parameter("dataset")

    if ds is None:
      raise ValueError('"dataset" parameter must be set.')

    new_min, new_max = self._get_parameter("min"), self._get_parameter("max")
    
    min_vals, max_vals = None, None

    dl = torch.utils.data.DataLoader(ds, batch_size=self.__batch_size)

    def print_progress(index):
      progress_bar(index,
        len(dl),
        pre_text=" Computing dimwise min/max ")

    print_progress(0)
    for i, (x, _) in enumerate(dl, 1):
      # min/max return pair of (values, indices per dim)
      # we only need the former
      batch_min, batch_max = torch.min(x,0)[0], torch.max(x,0)[0]

      if min_vals is None:
        min_vals = batch_min
      else:
        min_vals = torch.minimum(min_vals,batch_min)
      if max_vals is None:
        max_vals = batch_max
      else:
        max_vals = torch.maximum(max_vals,batch_max)

      print_progress(i)

    if new_min > new_max:
      raise ValueError('"min" must not be greater than "max".')

    # x' = (x - min(xs)) / (max(xs) - min(xs)) * (new_max - new_min) + new_min
    #    = x * scale + (new_min - min(xs) * scale)
    #  where scale = (new_max - new_min) / (max(xs) - min(xs))
    dval = (max_vals - min_vals)
    # if min(xs) == max(xs), x' should be in
    # [new_min, new_max] = [x - min(xs) + new_min, x - min(xs) + new_max]
    # we will choose x' = x - min(xs) + new_min + new_max/2
    # x * scale + (new_min - min(xs) * scale) == x - min(xs) + new_min + new_max/2
    self.__scale = (new_max - new_min) / dval
    self.__scale[dval == 0.] = 1.
    self.__shift = new_min - min_vals * self.__scale

  @staticmethod
  def __make_data_path(path):
    return join_paths(path, "_MinMaxNormalization__params.pkl")

  def _save(self, value, path):
    data = {
      "shift": self.__shift,
      "scale": self.__scale,
    }
    with open(self.__make_data_path(path), "wb") as file:
      pickle.dump(data, file)

    return super()._save(value, path)

  def _load(self, path):
    with open(self.__make_data_path(path), "rb") as file:
      data = pickle.load(file)
    self.__shift = data["shift"]
    self.__scale = data["scale"]

  def __len__(self):
    self.ensure_available()
    return len(self.__ds)

  def __call__(self, val):
    self.ensure_available()
    return val * self.__scale + self.__shift

class EncodingStretch(VirtualReproducible):
  """
  Takes as inputs data with a fixed range per dimension and translates them to
  a different shape by stretching or shrinking the encoding range to take up
  the respective multiple of space. I.e. an range of 2**8 becomes a range of
  2**16 on a shape with twice the number of dimensions and a range of 2**4 on
  half the number of dimensions.

  Inputs will be rounded after shrinking in case of dimensionality reduction.
  """
  def __init__(self, parameters={},
    **kwargs):
    """
    Args:
      parameters: Dictionary of parameters to set in the reproducible.
        Parameters for EncodingStretch are:
        - min: The minimum of the range that should be encoded per input
          dimension.
        - max: The maximum of the range that should be encoded.
        - output_shape: If not None, all outputs will be in that shape. If
          output shape is greater than input shape, input dimensions will be
          spread over multiple output dimensions in order. If output shape is
          smaller, input dimensions will be compressed into the same output
          dimension.
    """
    parameters = {**{
        "_EncodingStretch__version": "1.0.2",
        "min": 0.,
        "max": 256.,
        "output_shape": None
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__output_dim_count = np.prod(self._get_parameter("output_shape"))
    self.__range = self._get_parameter("max") - self._get_parameter("min")

  def __as_target_shape(self, x):
    x = x.reshape(x.shape[:1] + self._get_parameter("output_shape"))
    return x.type(self.__init_type)

  def __pack_dims(self, x, factor, pad):
    """
    Take an input with greater total size and transfer it to the smaller target
    dimensionality.

    parameters:
      factor: How many dimensions will be packed into one. Must be an integer.
      pad: Number of zero dimensions to pad to input.
    """
    x = torch.nn.functional.pad(x.flatten(start_dim=1), (0,pad))

    # reshape x such that last dimension will be reduced
    x = x.view(x.shape[0], self.__output_dim_count, -1)

    # put values into different parts of the range
    x = x.floor()
    for k in range(1, factor):
      x[:,:,factor - k] *= self.__range ** (k/factor)

    # sum parts into whole and return output
    return self.__as_target_shape(x.sum(-1))

  def __unpack_dims(self, x, factor, pad):
    """
    Take an input with smaller total size and transfer it to the greater target
    dimensionality.

    parameters:
      factor: Into how many dimensions each input dimension will be split.
      pad: Number of zero dimensions to add to output.
    """
    x = x.flatten(start_dim=1)

    batch_size, elem_count = x.shape

    def make_part(x, part):
      """
      extracts the part of x that will be encoded in the part'th position
      """
      # put x into correct position
      less_sig_parts = factor - part - 1
      x = (x / self.__range**less_sig_parts).floor()

      # remove part from more singificant positions
      return x - (x / self.__range).floor() * self.__range

    x = torch.cat([make_part(x,k) for k in range(factor)])

    # reorder x such that dimensions from the same input are next to each other
    # initially first (batch) contains parts in order
    x = x.view(batch_size,factor,elem_count).permute(0,2,1)

    # append padding and reshape to target shape
    x = torch.nn.functional.pad(x.flatten(start_dim=1), (0,pad))
    return self.__as_target_shape(x)

  def __scale_range(self, x, bit_factor):
    """
    Scales the range of values such that the new values would take up
    bit_factor as much space as before.
    """
    # assuming an initial range of 2^n
    # x * range ^ (bit_factor - 1)
    #  = x / range * range^bit_factor
    #  = x / range * (2^n)^bit_factor
    #  = x / range * 2^(n * bit_factor)
    # i.e. new range becomes 2^(n * bit_factor)
    # (calculation works independent of base)
    return x.double() * self.__range ** (bit_factor - 1)

  def __call__(self, val):
    """
    Reshape the input, assuming its first dimension is a batch dimension.
    """
    self.ensure_available()

    input_dim_count = np.prod(val.shape[1:])

    # change range to start at 0
    val = val - self._get_parameter("min")

    self.__init_type = val.dtype

    if self.__output_dim_count < input_dim_count:
      # reducing dimension
      old_dims_per_new = roundup_idiv(input_dim_count, self.__output_dim_count)

      val = self.__scale_range(val, 1 / old_dims_per_new)

      input_zero_dims = self.__output_dim_count*old_dims_per_new - input_dim_count

      return self.__pack_dims(val, old_dims_per_new, input_zero_dims)
    elif self.__output_dim_count > input_dim_count:
      # increasing dimension
      new_dims_per_old = self.__output_dim_count // input_dim_count

      val = self.__scale_range(val, new_dims_per_old)

      output_zero_dims = self.__output_dim_count - input_dim_count * new_dims_per_old

      return self.__unpack_dims(val, new_dims_per_old, output_zero_dims)
    else:
      # just reshape
      return self.__as_target_shape(val)
