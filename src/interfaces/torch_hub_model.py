import torch

from interfaces.simple_model import SimpleTorchModel

class TorchHubModel(SimpleTorchModel):
  """
  A model loaded from torch.hub.
  """
  # TODO: currently not deterministic
  def __init__(self, model=None, name="Torch Hub Model", parameters={}, **kwargs):
    """
    Create a new TorchHubModel reproducible by loading a model from a torch
    model hub, see https://pytorch.org/docs/1.7.0/hub.html.

    Args:
      parameters: Additional parameters to add to this network. In addition to
        the parameters from SimpleTorchModel, the following are applicable:
         - repo: Github repository or local directory to load models from.
           Default is "pytorch/vision:v0.8.1".
         - model: Model to load.
    """
    # merge existing parameters with defaults for parameters of the MiniResnet
    # default values will overwritten by existing values in parameters
    parameters = {
      **{
        "model": model,
        "repo": "pytorch/vision:v0.8.1",
        "optimizer": "SGD",
        "weight_decay": 1e-4,
        "lr": 0.01,
        "epochs": 10,
        "loss": "CrossEntropy",
        "_TorchHubModel__version": "1.0.0"
      },
      **parameters}

    super().__init__(parameters=parameters, name=name, **kwargs)

  def _initialize_model(self):
    self.model = torch.hub.load(self._get_parameter("repo"),
      self._get_parameter("model"))

    super()._initialize_model()
