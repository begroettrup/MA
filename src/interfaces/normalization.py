from .reproducible import VirtualReproducible

import torch

from .command_line import progress_bar

class FixedValueModel (VirtualReproducible):
  def __init__(self, val, parameters={}, **kwargs):
    parameters = {
      **{
        "_FixedValueModel__version": "1.0.1",
        "value": val
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__value = torch.tensor(self._get_parameter("value"))
    self.__value = self.__value.reshape((1,) + self.__value.shape)

  def __call__(self, x):
    shape, device, dtype = x.shape, x.device, x.dtype

    del x

    val = self.__value.repeat([shape[0]] + [1]*(len(self.__value.shape) - 1))
    return val.to(device=device, dtype=dtype)

def data_mean(dataset, batch_size=2048):
  return data_mean_std(dataset, batch_size)[0]

def data_mean_std(dataset, batch_size=2048):
  """
  Returns mean and standard deviation of the data.
  """
  dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
  batch_count = len(dl)

  progress_bar(0, batch_count,
    pre_text=" Inspecting Data "
  )

  n_batches = 1.
  mean_sum = 0.
  square_mean_sum = 0.

  for i, (inputs, _) in enumerate(dl, 1):
    inputs = inputs.to(dtype=torch.double)
    # this is not numerically ideal, but that's fine due to large batch sizes
    batch_percentage = len(inputs)/batch_size
    n_batches += batch_percentage
    mean_sum += inputs.mean(dim=0) * batch_percentage
    square_mean_sum += (inputs**2).mean(dim=0) * batch_percentage

    progress_bar(i, batch_count,
      pre_text=" Inspecting Data "
    )

  mean = mean_sum / n_batches
  square_mean = square_mean_sum / n_batches

  return mean, (square_mean_sum - mean**2).sqrt()
