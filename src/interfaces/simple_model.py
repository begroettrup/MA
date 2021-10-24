import torch
import torch.nn as nn
import functools
import operator
import math

from os.path import join as join_paths

import interfaces.torch_utils as utils
from interfaces.reproducible import CallableReproducible
from interfaces.torch_model import ReproducibleTorchModel
from interfaces.epochs_model import EpochsModel
from interfaces.model_parameters import WithOptimizerParam

from interfaces.command_line import progress_bar

from interfaces.losses import get_loss_aggregator, get_metric_aggregator

from interfaces.test_results import test_model

class SimpleTorchModel(EpochsModel,
                       WithOptimizerParam,
                       ReproducibleTorchModel):
  """
  A torch model which is trained on a dataset with labels in a straightforward
  manner using a given optimizer and loss function. The actual model should be
  stored in self.model.

  When the model is produced, _initialize_model() is called once, as is
  _initialize_training() afterwards. _initialize_model() should be implemented
  to set up the model for training and forward() should be implemented as
  usually. _deinitialize_model() is the tear-down equivalent of
  _initialize_training().

  _initialize_training() sets up the training procedure internally. Unless
  _train() is also overwritten, _initialize_training() should make a super
  call if overwritten, same in _deinitialize_training().

  _initialize_model() also constructs the optimizer for this model. Thus a
  super call should be made unless _train() is overwritten for a custom
  training procedure.
  """
  # TODO: currently not deterministic
  def __init__(self, parameters={}, *, use_cuda=False, loader_workers=2,
    **kwargs):
    """
    Create a new SimpleTorchModel reproducible.

    Args:
      loader_workers: Number of workers used for loading data. Set to 0 for
        single process data loading.
      use_cuda: If set, the model will be created and trained on gpu.
      parameters: Additional parameters to add to this network.
        Applicable parameters:
         - loss: Loss function for training and evaluating the model, one of
             "L1", "MSE", "CrossEntropy" or "EntropyChange" or a reproducible
             or picklable callable to calculate the loss.
         - optimizer: Name of the optimizer to use in training. One of "SGD",
             "RMSProp" or "Adam".
         - epochs: Number of epochs to train for.
         - input: Potentially reproducible list of (x,y) pairs
         - shuffle_data: Whether to shuffle the input data during training.
         - batch_size: Training and testing batch size.
    """
    # merge existing parameters with defaults for parameters of the
    # SimpleTorchModel
    # default values will overwritten by existing values in parameters
    parameters = {
      **{
        "epochs": 5,
        "input": None,
        "lr": 0.001,
        "loss": None,
        # TODO: seeding
        # "seed": torch.initial_seed(),
        "batch_size": 128,
        "shuffle_data": True,
        "_SimpleTorchModel__version": "2.1.0",
        "tracked_metrics": { "Loss" }
      },
      **parameters}

    self.__loader_workers = loader_workers
    self.__training_initialized = False
    self.__use_cuda = use_cuda
    super().__init__(parameters=parameters, **kwargs)

  @staticmethod
  def version():
    """
    Returns a version string of this class. Note that this isn't necessarily
    the same as the reproducible version parameter as it also reflects changes
    in the outer functionality and not just in the reproducible process.
    """
    return "2.2.0"

  def _initialize_training(self):
    """
    Perform setup for _train(). Will be called once before training.
    """
    input_ = self._get_parameter("input")

    batch_size = self._get_parameter("batch_size")

    self.__trainloader = torch.utils.data.DataLoader(input_,
        batch_size=batch_size,
        shuffle=self._get_parameter("shuffle_data"),
        num_workers=self.__loader_workers)

    self.model.train()

  def _deinitialize_training(self):
    """
    Cleanup after _train(). Will be called once after training.
    """
    self.__trainloader = None
    self.model.eval()

  def _train(self):
    """
    Train for one epoch.
    """
    epoch = self._get_parameter("epochs")

    loss_aggregator = get_loss_aggregator(self._get_parameter("loss"))

    metric_aggregators = self.__get_aggregators(
      self._get_parameter("tracked_metrics"))

    for i, (inputs, labels) in enumerate(self.__trainloader, 1):
      self.__optimizer.zero_grad()
      inputs, labels = self.__on_correct_device(inputs, labels)

      outputs = self.model(inputs)

      post_text = ""
      for agg in metric_aggregators:
        with torch.no_grad():
          agg.batch_loss(outputs, labels)
        post_text += " | {}: {:6.3f}".format(
          agg.name, agg.full_loss())

      loss = loss_aggregator.batch_loss(outputs, labels)
      loss.backward()
      self.__optimizer.step()

      progress_bar(i, len(self.__trainloader),
        pre_text=" Epoch {} ".format(epoch),
        post_text=post_text
      )
      del loss, outputs, inputs, labels

    self.metrics = []
    for agg in metric_aggregators:
      self.metrics.append(agg.full_loss())

  def _initialize_model(self):
    if self.__use_cuda:
      self.model.cuda()
    self.__optimizer = self._get_optimizer(self.model.parameters())

  def __on_correct_device(self, /, *args):
    return utils.on_correct_device(self.__use_cuda, *args)

  def __perform_test(self, testset, aggregators):
    """
    Perform a test on the given data set.
    """
    return test_model(
      self.model, testset, aggregators,
      self._get_parameter("batch_size"),
      loader_workers=self.__loader_workers,
      use_cuda=self.__use_cuda)

  def __get_aggregators(self, metrics):
    metrics = metrics.copy()

    aggregators = []

    for metric in metrics:
      if metric == "Loss":
        agg = get_metric_aggregator(self._get_parameter("loss"))
        agg.name = "Loss"
      else:
        agg = get_metric_aggregator(metric)
      aggregators.append(agg)

    return aggregators

  def test(self, testset, metrics=["Loss"]):
    """
    Test the model on a testset.

    Args:
      testset: List-like of (x,y) pairs for testing on.
      metrics: List of goodness measures that should be computed for the model.
        May include:
        - "Loss": Will calculate the test data loss.
        - "Accuracy": Will assume the model output is a one-of representation
          of a class and that the labels are integer class labels.
        - any valid loss function (see "loss" parameter)
    Return: List of the values of each metric in the same order as metrics
      input list.
    """
    self.ensure_available()
    
    aggregators = self.__get_aggregators(metrics)

    return self.__perform_test(testset, aggregators)

  @staticmethod
  def __make_representation_path(path):
    return join_paths(path, "_SimpleModel__state")

  def _save(self, value, path):
    state = {
      'optimizer': self.__optimizer.state_dict()
    }
    try:
      state['metrics'] = self.metrics
    except AttributeError:
      pass
    torch.save(state, self.__make_representation_path(path))
    super()._save(value, path)

  def _unloadable(self):
    return True

  def unload(self):
    try:
      del self.__optimizer, self.model
    except AttributeError:
      pass

    super().unload()

  def _load(self, path):
    ret = super()._load(path)
    state = torch.load(self.__make_representation_path(path))
    self.__optimizer.load_state_dict(state['optimizer'])
    try:
      self.metrics = state['metrics']
    except KeyError:
      pass
    return ret

  def __call__(self, *args, **kwargs):
    self.ensure_available()
    return self.model(*self.__on_correct_device(*args), **kwargs)
