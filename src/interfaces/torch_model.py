from interfaces.reproducible import Reproducible
import torch
from os.path import join as join_paths

class ReproducibleTorchModel(Reproducible):
  """
  More specialized version of Reproducible that implements a more disk space
  efficient save and load mechanism for classes that are mainly represented by
  a torch model in their self.model attribute.
  """
  @staticmethod
  def __make_representation_path(path):
    return join_paths(path, "_ReproducibleTorchModel__model")

  def _initialize_model(self):
    """
    This is called when loading and should be used to initialize the model
    into a state corresponding to the parameters.
    """
    pass

  def _save(self, value, path):
    torch.save(self.model.state_dict(), self.__make_representation_path(path))
    super()._save(value, path)

  def _load(self, path):
    self._initialize_model()
    self.model.load_state_dict(torch.load(self.__make_representation_path(path)))
    # set to evaluation mode when training is done.
    if self._reproduction_depth() == 1:
      self.model.eval()
    return super()._load(path)

  def __call__(self, *args, **kwargs):
    self.ensure_available()
    return self.model(*args, **kwargs)
