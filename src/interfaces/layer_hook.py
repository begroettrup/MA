import torch

from .reproducible import VirtualReproducible

class LayerHook(VirtualReproducible):
  """
  A model which instead of producing it's usual outputs produces the configured
  output.
  """
  def __init__(self, model, layers, parameters={}, **kwargs):
    """
    Args:
      layers: Layername or iterable of layer names in order of extraction.
    """
    parameters = {**{
        "_LayerHook__version": "1.0.0",
        "model": model,
        "layer": layers,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    def save_output(module, input, output):
      self.__last_output = output

    layers = self._get_parameter("layer")

    if type(layers) == str:
      layers = [layers]

    self.__model = self._get_parameter("model")

    def install_hook(repr_model):
      mod = repr_model.model

      for l in layers:
        mod = mod._modules[l]

      mod.register_forward_hook(save_output)

    self.__model.once_available(install_hook)

  def __call__(self, *args, **kwargs):
    self.ensure_available()
    self.__model(*args, **kwargs)

    output = self.__last_output

    if not torch.is_grad_enabled():
      if type(output) == tuple:
        output = output[0].detach(), *output[1:]
      else:
        output = output.detach()

    del self.__last_output
    return output
