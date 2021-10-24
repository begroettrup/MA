from interfaces.model_parameters import WithOptimizerParam
from interfaces.reproducible import VirtualReproducible

from interfaces.command_line import progress_bar
from interfaces.losses import get_loss_aggregator, FixedExponentLoss

import torch
import interfaces.torch_utils as tut

class ZeroSuggestionModel(VirtualReproducible):
  def __init__(self, input_shape, parameters={}, **kwargs):
    parameters = {
      **{
        "input_shape": input_shape,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def __call__(self, y):
    return torch.zeros((y.shape[0],) + self._get_parameter("input_shape"))

class GradientDescentInverter(VirtualReproducible, WithOptimizerParam):
  """
  Inverts a function by performing gradient descent on it.
  """
  def __init__(self, function, suggester, parameters={}, print_progress=False,
    use_cuda=False, **kwargs):
    parameters = {
      **{
        "max_iterations": 1000,
        "function": function,
        "suggester": suggester,
        # tuple of minimum factor from old to new and the number of steps that
        # it needs to be achieved in to not break
        "early_stopping_params": None,
        "target_loss": None,
        "loss": "L1",
        "lr": .1,
        # output will be transformed with this function
        "activation": None,
        "regularization_coefficient": 0.,
        # used in regularization to scale the element to image scale
        "regularization_norm_factor": 1/256.,
        "optimizer": "Adagrad",
        "_GradientDescentInverter__version": "1.1.0",

      },
      **parameters}

    self.__print_progress = print_progress
    self.__use_cuda = use_cuda

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__suggester = self._get_parameter("suggester")

  def __call__(self, y):
    self.ensure_available()
    with torch.no_grad():
      var = torch.autograd.Variable(
        *tut.on_correct_device(self.__use_cuda, self.__suggester(y)),
        requires_grad=True)

    stop_params = self._get_parameter("early_stopping_params")
    
    target_loss = self._get_parameter("target_loss")

    if stop_params is not None:
      stop_factor, loss_buffer = stop_params
      old_losses = []

    optim = self._get_optimizer([var])

    loss_aggregator = get_loss_aggregator(self._get_parameter("loss"))

    f = self._get_parameter("function")

    iters = self._get_parameter("max_iterations")

    activation = self._get_parameter("activation")

    reg_coeff = self._get_parameter("regularization_coefficient")

    if reg_coeff != 0.:
      norm_factor = self._get_parameter("regularization_norm_factor")
      reg_loss = FixedExponentLoss(1, 1)

    def regularization_term(y):
      if reg_coeff == 0.:
        return 0.
      y_unscaled = y / norm_factor
      return reg_coeff * reg_loss(y, y_unscaled.round() * norm_factor)

    for i in range(iters):
      optim.zero_grad()

      with tut.grad():
        x = var

        if activation is not None:
          x = activation(x)

        loss = loss_aggregator.batch_loss(y, f(x)) + regularization_term(f(x))

        loss.backward()
        optim.step()

      if self.__print_progress:
        post_text = " | Loss: {:6.3f}".format(loss)

        progress_bar(i + 1, iters,
          pre_text=" Inverting ",
          post_text=post_text
        )

      if stop_params is not None:
        old_losses.append(loss.detach())
        if len(old_losses) > loss_buffer:
          old_losses = old_losses[1:]

        if len(old_losses) == loss_buffer:
          impr_factor = old_losses[-1] / old_losses[0]
          if impr_factor > stop_factor:
            if self.__print_progress:
              print()
            break

      if target_loss is not None:
        if loss < target_loss:
          if self.__print_progress:
            print()
          break

    return x.detach()
