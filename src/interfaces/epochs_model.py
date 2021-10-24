from .reproducible import Reproducible
from .named import Named

class EpochsModel(Reproducible, Named):
  """
  A reproducible model which is trained over a number of epochs.

  When the model is produced, _initialize_model() is called once, as is
  _initialize_training() afterwards. Contrary to _initialize_model(),
  _initialize_training() is also called when training is continued on an
  existing version of the model, e.g. after it was loaded from disk or used
  with a different epoch setting.

  _deinitialize_model() is the tear-down equivalent of _initialize_training().
  It is called once after each set of training.

  The training method should be implemented in _train() and will be called once
  per epoch.
  """
  # TODO: currently not deterministic
  def __init__(self, parameters={}, **kwargs):
    """
    Create a new EpochsModel reproducible.

    Args:
      name: Name to identify the model with in status messages.
      parameters: Additional parameters to add to this network.
        Applicable parameters:
         - epochs: Number of epochs to train for.
    """
    # merge existing parameters with defaults for parameters of the
    # EpochsModel
    # default values will overwritten by existing values in parameters
    parameters = {
      **{
        "epochs": None,
        "_EpochsTorchModel__version": "0.1.2"
      },
      **parameters}

    self.__training_initialized = False
    super().__init__(parameters=parameters, **kwargs)

  def _initialize_model(self):
    """
    Set up this object such that forward() is usable. Should be overwritten by
    subclasses.
    """
    raise NotImplementedError()

  def _initialize_training(self):
    """
    Perform setup for _train(). Will be called once before training.
    """
    raise NotImplementedError()

  def _deinitialize_training(self):
    """
    Cleanup after _train(). Will be called once after training.
    """
    raise NotImplementedError()

  def _train(self):
    """
    Cleanup after _train(). Will be called once after training.
    """
    raise NotImplementedError()

  def _delete_cache(self):
    if not self._is_reproducing():
      self.__training_initialized = False
    super()._delete_cache()

  def _produce(self):
    if self._get_parameter("epochs") is None:
      raise InputError(self.personalized_message(
        '"epochs" parameter must be set',
        " for {name}",
        None,
        '.', separator=""))

    if self._get_parameter("epochs") <= 0:
      self._initialize_model()
    else:
      self._set_parameter("epochs", self._get_parameter("epochs") - 1)
      self.reproduce_value()
      self._set_parameter("epochs", self._get_parameter("epochs") + 1)
      if not self.__training_initialized:
        self._initialize_training()
        print(self.personalized_message("== Training", "{name}", "Epochs Model", "=="))
        self.__training_initialized = True
      self._train()
    # set to evaluation mode when training is done.
    if self._reproduction_depth() == 1:
      if self.__training_initialized:
        self._deinitialize_training()
        self.__training_initialized = False
