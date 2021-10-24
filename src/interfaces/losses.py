import torch
import operator
import math

from .reproducible import VirtualReproducible
from . import torch_utils as utils

def large_mean(x):
  """
  Calculates the mean of x using a method that works even for values near the
  floating point maximum.
  """
  return (x / len(x)).sum()

def log_mean_exp(x):
  """
  Returns the logarithm of the mean of e to the power of x.
  """
  # doesn't use mean() to avoid exceeding floating point maximum in sum
  # (torch first takes the sum and then divides through the count)
  return large_mean(x.exp()).log()

def same_rounding_chance(dx):
  """
  This represents the chance that two points with dimension wise distance of dx
  are rounded to the same point assuming a uniformly distributed distance from
  the next rounding point.

  This loss might be problematic as it:
   - becomes very small with high dimension (expected value goes to 0)
   - is 0 with 0 gradient if any dimension is >= 1
  """
  return torch.maximum(1-dx,torch.tensor(0.)).prod()

def entropy_change(outputs, labels):
  """
  Returns log E_q[e^f(x)] - Eₚ[f(x)] where q is the reference and p the target distribution
  assuming the reference distribution is labeled with 1 and the target distribution is
  labeled with 0.

  This is equivalent to the change in cross-entropy introduced by reweighing the reference
  distribution according to e^f(x) using the formula p'(x)=q(x)e^f(x) / E_q[e^f(x)].
  """
  out_q = outputs[labels == 1]
  out_p = outputs[labels == 0]
  q_part = log_mean_exp(out_q) if len(out_q) > 0 else 0.
  p_part = out_p.mean() if len(out_p) > 0 else 0.
  loss = q_part - p_part
  return loss

def log_mean(x):
  """
  Returns the logarithm of the mean of the values.
  """
  # doesn't use mean() to avoid exceeding floating point maximum in sum
  # (torch first takes the sum and then divides through the count)
  return large_mean(x).log()

def entropy_change_noexp(outputs, labels):
  """
  Same as entropy_change() but directly uses the original output as weights instead of taking
  the exponent first.

  Returns log E_q[f(x)] - Eₚ[log f(x)] where q is the reference and p the target distribution
  assuming the reference distribution is labeled with 1 and the target distribution is
  labeled with 0.

  This is equivalent to the change in cross-entropy introduced by reweighing the reference
  distribution according to f(x) using the formula p'(x)=q(x)f(x) / E_q[f(x)].
  """
  out_q = outputs[labels == 1]
  out_p = outputs[labels == 0]
  q_part = log_mean(out_q) if len(out_q) > 0 else 0.
  p_part = out_p.log().mean() if len(out_p) > 0 else 0.
  # first term doesn't use mean() to avoid exceeding floating point maximum in sum
  # (torch first takes the sum and then divides through the count)
  loss = q_part - p_part
  return loss

class LossFunctionAggregate:
  """
  A loss function of a neural network that can be sensefully estimated via
  aggregation during training but may require all data for an exact
  computation.

  This class takes care of aggregating values, computing batchwise losses and
  returning an estimate for the aggregated loss.
  """
  def __init__(self, /,
    name, loss_function,
    loss_transform=lambda x: x,
    expansion=operator.mul,
    accumulator=operator.add,
    reduction=operator.truediv,
    result_transform=lambda x: x,
    init_value=0.,
    aggregation_is_exact=True):
    """
    Args:
      name: Name of the loss function.
      loss_function: Loss function which receives a batch of data and a batch
        of labels as inputs to produce a single loss value represented by a
        1 element tensor.
      loss_transform: Transformation of the batch loss before
        backpropagation. The aggregation function will receive the loss as
        returned by this transformation. During batch-wise descent algorithms
        the transformed loss will serve as target for optimizations.
        Should return a tensor containing a single value such that 
        backpropagation can be performed correctly.
      expansion: Create a per-sample equivalent of the batch-wise transformed
        loss such that the reduction will be able to correctly scale down the
        value.
        It will receive the value of the transformed loss as first argument
        and the number of elements in the current sample as second.
      accumulator: Function to accumulate loss over multiple batches. It will
        receive the current accumulation result as first argument and the
        output of expansion as second.
      reduction: Function to transform the accumulated result depending on
        the number of inputs. Receives the accumulation as first argument
        and the number of elements that have been accumulated as second.
      result_transform: Transformation that transforms the result of the
        accumulation before display and before consideration as loss during
        testing. It receives the accumulation result as a value as argument.
        This function and reduction are always performed right after another
        without the intermediate result being used or returned elswhere.
        It is unnecessary to set both of them safe for convenience.
      init_value: Initial value for the accumulation result.
      aggregation_is_exact: Flag to signal whether the result of the
        accumulation is the same as the loss over the full set. This can be
        used to decide whether memory efficient batch loss accumulation is
        possible in testing or whether all elements have to be passed to the
        function at once.
    """
    self.name = name
    self.__loss_function = loss_function
    self.__loss_transform = loss_transform
    self.__accumulator = accumulator
    self.__result_transform = result_transform
    self.__init_value = init_value
    self.__aggregation_is_exact = aggregation_is_exact
    self.__expansion = expansion
    self.__reduction = reduction
    self.reset_accumulation()

  def aggregation_is_exact(self):
    """
    Returns whether the accumulation of batch results is mathematically equal
    to the loss of the full data set (not considering numerical
    imprecisions).
    """
    return self.__aggregation_is_exact

  def reset_accumulation(self):
    """
    Discard the current accumulation results and reset to the initial value.

    Should be called before every loop.
    """
    self.__accumulated_value = self.__init_value
    self.__number_elements = 0

  def raw_loss(self, outputs, labels):
    """
    Returns the unmodified loss for the given batch or dataset. This should
    be used when testing on the whole data set or using full-data gradient
    descent.

    Args:
      outputs: Outputs of the model for the current batch. Should be a tensor
        of outputs per inputs.
      labels: Labels corresponding to the outputs. Labels are directly passed
        to the loss function and don't need to satisfy any additional
        constraints.
    """
    return self.__loss_function(outputs, labels)

  def batch_loss(self, outputs, labels):
    """
    Returns the loss for the current batch that should be used in batch-wise
    gradient descent.

    This function will also accumulate the loss of the batch.

    Args:
      outputs: Outputs of the model for the current batch. Should be a tensor
        of outputs per inputs.
      labels: Labels corresponding to the outputs. Labels are directly passed
        to the loss function and don't need to satisfy any additional
        constraints.
    """
    self.__number_elements += len(outputs)

    raw_loss = self.raw_loss(outputs, labels)

    curr_loss = self.__loss_transform(raw_loss)
    with torch.no_grad():
      self.__accumulated_value = self.__accumulator(
        self.__accumulated_value,
        self.__expansion(curr_loss.item(), len(outputs)))

    # We return the loss transformed by some strictly increasing function,
    # which is equivalent in regards to global optimization but a closer
    # approximation for the full loss in batch optimization
    # This may be changed if that approach proves problematic for some reason
    # In the case of an exponential transformation, this leads to a smaller
    # gradient by an exponential factor for small losses.
    return curr_loss

  def aggregation_result(self):
    """
    Returns the result of the aggregation after all transformations. This is
    an estimate of the total data loss but may be a mathematically (as
    opposed to just numerically) different if self.aggregation_is_exact()
    is False.
    """
    return self.__result_transform(self.__reduction(
      self.__accumulated_value, self.__number_elements))

  def full_loss(self):
    """
    Returns the actual loss over all elements since last reset of accumulation.

    To be implemented by subclasses in case the aggreagtion result ist not
    inherently exact.
    """
    if self.aggregation_is_exact():
      return self.aggregation_result()
    else:
      raise NotImplementedError()

class EntropyChangeAggregator(LossFunctionAggregate):
  """
  A spezialized aggregator for the entropy change loss that also tracks the
  real full data loss.
  """
  def __init__(self):
    self.__reset()

    super().__init__(
      "Entropy Change", entropy_change,
      loss_transform=torch.exp,
      result_transform=math.log,
      aggregation_is_exact=False)

  def __reset(self):
    self.__exp_q_mean = 0.
    self.__q_count = 0
    self.__p_mean = 0.
    self.__p_count = 0

  def reset_accumulation(self):
    self.__reset()
    super().reset_accumulation()

  def batch_loss(self, outputs, labels):
    with torch.no_grad():
      out_q = outputs[labels == 1]
      out_p = outputs[labels == 0]
      # accumulating for q_part = large_mean(out_q.exp()).log()
      if len(out_q) > 0:
        self.__exp_q_mean, self.__q_count = utils.accumulate_mean(
          self.__exp_q_mean, self.__q_count, large_mean(out_q.exp()), len(out_q))
      # accumulating for p_part = out_p.mean()
      if len(out_p) > 0:
        self.__p_mean, self.__p_count = utils.accumulate_mean(
          self.__p_mean, self.__p_count, out_p.mean(), len(out_p))
    return super().batch_loss(outputs, labels)

  def full_loss(self):
    with torch.no_grad():
      q_part = self.__exp_q_mean.log()
      p_part = self.__p_mean
      return (q_part - p_part).item()

class DirectEntropyChangeAggregator(LossFunctionAggregate):
  """
  A spezialized aggregator for the direct entropy change loss that also tracks
  the real full data loss.
  """
  def __init__(self):
    self.__reset()

    super().__init__(
      "Direct Entropy Change", entropy_change_noexp,
      loss_transform=torch.exp,
      result_transform=math.log,
      aggregation_is_exact=False)

  def __reset(self):
    self.__q_mean = 0.
    self.__q_count = 0
    self.__log_p_mean = 0.
    self.__p_count = 0

  def reset_accumulation(self):
    self.__reset()
    super().reset_accumulation()

  def batch_loss(self, outputs, labels):
    with torch.no_grad():
      out_q = outputs[labels == 1]
      out_p = outputs[labels == 0]
      # accumulating for q_part = large_mean(out_q).log()
      if len(out_q) > 0:
        self.__q_mean, self.__q_count = utils.accumulate_mean(
          self.__q_mean, self.__q_count, large_mean(out_q), len(out_q))
      # accumulating for p_part = out_p.log().mean()
      if len(out_p) > 0:
        self.__log_p_mean, self.__p_count = utils.accumulate_mean(
          self.__log_p_mean, self.__p_count, out_p.log().mean(), len(out_p))
    return super().batch_loss(outputs, labels)

  def full_loss(self):
    with torch.no_grad():
      q_part = self.__q_mean.log()
      p_part = self.__log_p_mean
      return (q_part - p_part).item()

def l1_aggregator():
  """
  Return an aggregator for measuring median absolute error.
  """
  return LossFunctionAggregate("L1", torch.nn.L1Loss())

def l2_aggregator():
  """
  Return an aggregator for measuring mean squared error.
  """
  return LossFunctionAggregate("L2", torch.nn.MSELoss())

def entropy_change_aggregator():
  return EntropyChangeAggregator()

def direct_entropy_change_aggregator():
  return DirectEntropyChangeAggregator()

def cross_entropy_aggregator():
  """
  Return an aggregator for measuring cross entropy.
  """
  return LossFunctionAggregate("Cross Entropy", torch.nn.CrossEntropyLoss())

def accuracy_aggregator():
  """
  Returns an aggregator for measuring model accuracy.
  """
  return LossFunctionAggregate("Accuracy", utils.accuracy)

def get_loss_aggregator(loss, name=None):
  """
  Returns the aggregator corresponding to a loss by name of that loss or a
  function that respresents that loss.

  Possible losses are:
    - "L1" aka "MAE"
    - "MSE" aka "L2"
    - "EntropyChange"
    - "EntropyChange NoExp" aka "Direct EntropyChange"
    - "CrossEntropy"
  """
  if type(loss) == str:
    if loss in ["L1", "MAE"]:
      return l1_aggregator()
    elif loss in ["MSE", "L2"]:
      return l2_aggregator()
    elif loss in ["EntropyChange"]:
      return entropy_change_aggregator()
    elif loss in ["EntropyChange NoExp", "Direct EntropyChange"]:
      return direct_entropy_change_aggregator()
    elif loss in ["CrossEntropy"]:
      return cross_entropy_aggregator()
    else:
      raise ValueError("Unknown loss function: " + loss)
  else:
    return LossFunctionAggregate("Custom Loss" if name is None else name, loss)

def get_metric_aggregator(metric, name=None):
  """
  Returns the aggregator corresponding to a testing metric by the name of that metric.
  This may be any loss aggregator or:
  - "Accuracy"
  """
  try:
    return get_loss_aggregator(metric, name)
  except ValueError:
    if metric == "Accuracy":
      return accuracy_aggregator()
    else:
      raise ValueError("Unknown metric: " + metric)

class ReproducibleLoss(VirtualReproducible):
  def __init__(self, name, parameters={}, **kwargs):
    parameters = {
      **{
        "name": name,
        "_ReproducibleLoss__version": "1.0.0",
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    loss = self._get_parameter("name")

    if loss in ["L1", "MAE"]:
      self.loss = torch.nn.L1Loss()
    elif loss in ["MSE", "L2"]:
      self.loss = torch.nn.MSELoss()
    elif loss in ["EntropyChange"]:
      self.loss = torch.nn.L1Loss()
    elif loss in ["CrossEntropy"]:
      return torch.nn.CrossEntropyLoss()
    else:
      raise ValueError("Unknown loss function: " + loss)

  def __call__(self, x1, x2):
    self.ensure_available()
    return self.loss(x1, x2)

class FixedExponentNorm:
  def __init__(self, exponent, ord=2):
    self.q = exponent
    self.ord = ord
    
  def __eq__(self, other):
    if isinstance(other, FixedExponentNorm):
      return self.q == other.q and self.ord == other.ord
    else:
      return False

  def __call__(self, x):
    x = x.view(len(x),-1)

    if self.ord == float('inf'):
      return x.abs().max(dim=1)**self.q
    elif self.ord == 0:
      return (x != 0).sum(dim=1)**self.q
    else:
      return (x.abs() ** self.ord).sum(dim=1)**(self.q / self.ord)

class FixedExponentLoss:
  def __init__(self, exponent, ord=2):
    self.norm = FixedExponentNorm(exponent, ord)

  def __call__(self, x1, x2):
    return self.norm(x1 - x2).mean()

  def __eq__(self, other):
    if isinstance(other, FixedExponentLoss):
      return self.norm == other.norm
    else:
      return False
