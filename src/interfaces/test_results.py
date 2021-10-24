from .reproducible import Reproducible

from . import losses as ilosses

from .command_line import progress_bar

from .torch_utils import on_correct_device

import torch

def test_model(model, testset, aggregator, batch_size=128, loader_workers=2,
  use_cuda=False):
  """
  Tests a given model on the given aggregator(s).
  """
  was_single_aggregator = False
  if type(aggregator) != list:
    try:
      aggregators = list(aggregator)
    except TypeError:
      aggregators = [aggregator]
      was_single_aggregator = True
  else:
    aggregators = aggregator

  testloader = torch.utils.data.DataLoader(testset,
      batch_size=batch_size, num_workers=loader_workers)
  test_batch_count = len(testloader)

  curr_elem = 0

  for i, (inputs, labels) in enumerate(testloader, 1):
    inputs, labels = on_correct_device(use_cuda, inputs, labels)

    end_elem = curr_elem + len(inputs)

    with torch.no_grad():
      outputs = model(inputs)

    curr_elem = end_elem

    post_text = ""

    for agg in aggregators:
      agg.batch_loss(outputs, labels)
      post_text += " | {}: {:6.3f}".format(
        agg.name, agg.full_loss())

    progress_bar(i, test_batch_count,
      pre_text=" Testing ",
      post_text=post_text
    )

    del outputs

  if was_single_aggregator:
    return aggregators[0].full_loss()
  else:
    results = []
    for agg in aggregators:
      results.append(agg.full_loss())

    return results

class ReproducibleModelTest(Reproducible):
  def __init__(self, model, testset, metrics, metric_names=None,
      batch_size=128, loader_workers=2, use_cuda=False, parameters={}, **kwargs):
    parameters = {
      **{
        "_ReproducibleModelTest__version": "1.0.0",
        "model": model,
        "testset": testset,
        "_TestResult__model_version": None,
      },
      **parameters}

    self.__batch_size=batch_size
    self.__loader_workers=loader_workers
    self.__use_cuda = use_cuda

    self.metric_params = []

    if type(metrics) == dict:
      for metric_name, metric in metrics.items():
        if metric_name in parameters.keys():
          raise ValueError("Metric name '" + metric_name + "' is already a parameter!")

        self.metric_params.append(metric_name)
        parameters[metric_name] = metric
      self.metric_params.sort()
    else:
      for i, metric in enumerate(metrics):
        metric_name = "Metric {}".format(i)
        self.metric_params.append(metric_name)
        parameters[metric_name] = metric

    super().__init__(parameters=parameters, **kwargs)

  def _set_dependent_parameters(self):
    try:
      self._set_parameter("_TestResult__model_version",
                          self._get_parameter("model").version())
    except AttributeError:
      pass
    return super()._set_dependent_parameters()

  def _produce(self):
    aggs = []

    for metric_name in self.metric_params:
      aggs.append(ilosses.get_metric_aggregator(self._get_parameter(metric_name),
                  metric_name))

    return test_model(
      self._get_parameter("model"),
      self._get_parameter("testset"),
      aggs,
      batch_size=self.__batch_size,
      loader_workers=self.__loader_workers,
      use_cuda=self.__use_cuda
    )

class TestResults(Reproducible):
  """
  Results of testing a model.
  """
  def __init__(self, /, parameters={}, **kwargs):
    """
    Create a new TestResults objects.

    Args:
      parameters: Dictionary of additional parameters. Behavior changing
        parameters are:
        - model: Model to test, should have a test(testset, metrics) function
        - metrics: Metrics to test with, valid parameters are according to the
          test() function of the model.
        - testset: List-like dataset of (x,y) pairs to test with.
    """
    parameters = {
      **{
        "_TestResults__version": "1.1.0",
        "model": None,
        "metrics": ["Loss"],
        "testset": None,
        "_TestResult__model_version": None,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _set_dependent_parameters(self):
    try:
      self._set_parameter("_TestResult__model_version",
                          self._get_parameter("model").version())
    except AttributeError:
      pass
    return super()._set_dependent_parameters()

  def _produce(self):
    model = self._get_parameter("model")
    return model.test(
      self._get_parameter("testset"),
      metrics=self._get_parameter("metrics"))
