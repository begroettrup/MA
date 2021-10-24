from interfaces.parametrizable import Parametrizable

import torch

def get_optimizer_by_name(name, model_parameters, lr=0.01, weight_decay=0., **kwargs):
  """
  Returns an optimizer from a name and some common parameters.

  Args:
    model_parameters: All optimizable parameters of the model.
    lr: Learning rate for the optimizer.
    name: Name of the optimizer. Any of:
      - "SGD"
      - "SGD NAG"
      - "SGD Nesterov"
      - "RMSProp"
      - "Adam"
      - "Adamax"
  """
  if "SGD" in name:
    nesterov = "NAG" in name or "Nesterov" in name
    return torch.optim.SGD(
      model_parameters, lr=lr,
      weight_decay=weight_decay,
      nesterov=nesterov, **kwargs)
  elif name == "RMSProp":
    return torch.optim.RMSprop(
      model_parameters, lr=lr,
      weight_decay=weight_decay, **kwargs)
  elif name == "Adam":
    return torch.optim.Adam(
      model_parameters, lr=lr,
      weight_decay=weight_decay, **kwargs)
  elif name == "Adagrad":
    return torch.optim.Adagrad(
      model_parameters, lr=lr,
      weight_decay=weight_decay, **kwargs)
  elif name == "Adamax":
    return torch.optim.Adamax(
      model_parameters, lr=lr,
      weight_decay=weight_decay, **kwargs)
  else:
    raise ValueError("Unknown optimizer " + name)

class WithOptimizerParam(Parametrizable):
  """
  Adds parameters to the parametrizable for torch optimizers.
  """
  def __init__(self, parameters={}, use_cuda=torch.cuda.is_available(), **kwargs):
    parameters={
      **{
        "optimizer": None,
        "lr": 0.01,
        "_OptimizerParam__version": "2.0.0",
        "weight_decay": 0.,
        "optimizer_kwargs": {}
      },
      **parameters}

    self.__use_cuda = use_cuda

    super().__init__(parameters=parameters, **kwargs)

  def _get_optimizer(self, model_parameters):
    """
    Returns an optimizer on the given model parameters according to the options
    set in the reproducible.
    """
    return get_optimizer_by_name(
        name=self._get_parameter("optimizer"),
        model_parameters=model_parameters,
        lr=self._get_parameter("lr"),
        weight_decay=self._get_parameter("weight_decay"),
        **self._get_parameter("optimizer_kwargs")
      )
