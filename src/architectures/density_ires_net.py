import sys

sys.path.append("../3rdparty/invertible-resnet")

import torch
import numpy as np
import torch.nn as nn

from .general_iresnet import IResNet as BaseIResNet

import models.utils_cifar as utils_iResNet

from os.path import join as join_paths

from interfaces.epochs_model import EpochsModel
from interfaces.model_parameters import WithOptimizerParam
from interfaces.command_line import progress_bar

import interfaces.torch_utils as utils

class DensityIResNet(EpochsModel, WithOptimizerParam):
  """
  An invertible ResNet for density estimation, trained for minimum bits per
  pixel.
  """
  def __init__(self, parameters={}, *,
      use_cuda=torch.cuda.is_available(),
      loader_workers=2,
      max_sample_iters=10,
      **kwargs):
    """
    Args:
        loader_workers: Number of workers used for loading data. Set to 0 for
          single process data loading.
        use_cuda: Whether to use cuda for model training and inference. There
          are no separate switches atm.
        max_sample_iters: Maximum amount of loops for fixed point iteration in
          inverse during sampling.
    """
    parameters={
      **{
        "input": None,
        # overwriting optimizer params for different defaults
        "lr": 0.1,
        "optimizer": "Adamax",
        "n_blocks": [4, 4, 4],
        "strides": [1, 2, 2],
        "n_channels": [16, 64, 256],
        "epochs": 200,
        "warmup_epochs": 10,
        # contraction coefficient for linear layers
        "coeff": .9,
        "initial_downsampling": 2,
        # initial injective padding
        # upsamples sample dimension by padding zeros
        "injective_padding": 0,
        # number of samples used for trace estimation
        "num_trace_samples": 1,
        # number of samples used in power series for matrix log
        "num_series_terms": 1,
        # number of power iterations used for spectral norm
        "num_power_iter": 5,
        # batch size for training, sampling and testing
        "batch_size": 32,
        # number of samples to initialize actnorm parameters with
        "init_batch": 1024,
        "actnorm": True,
        # "relu", "elu", "sorting", "softplus"
        "nonlin": "relu",
        "learn_prior": True,
        "_DensityIResNet__version": "1.0.0",
        "shuffle_data": True,
      },
      **parameters}

    self.__use_cuda = use_cuda
    self.__loader_workers = loader_workers
    self.__max_sample_iters = max_sample_iters

    super().__init__(parameters=parameters, **kwargs)

  @staticmethod
  def version():
    """
    Returns a version string of this class. Note that this isn't necessarily
    the same as the reproducible version parameter as it also reflects changes
    in the outer functionality and not just in the reproducible process.
    """
    return "1.0.1"

  def _initialize_model(self):
    input_ = self._get_parameter("input")
    input_shape = input_[0][0].shape

    self.model = BaseIResNet(
      in_shape=input_shape,
      n_blocks=self._get_parameter("n_blocks"),
      strides=self._get_parameter("strides"),
      n_channels=self._get_parameter("n_channels"),
      init_ds=self._get_parameter("initial_downsampling"),
      inj_pad=self._get_parameter("injective_padding"),
      coeff=self._get_parameter("coeff"),
      numTraceSamples=self._get_parameter("num_trace_samples"),
      numSeriesTerms=self._get_parameter("num_series_terms"),
      n_power_iter=self._get_parameter("num_power_iter"),
      actnorm=self._get_parameter("actnorm"),
      nonlin=self._get_parameter("nonlin"),
    )

    self.__setup_model()

    self.__optimizer = self._get_optimizer(self.model.parameters())

  def __setup_model(self):
    """
    Finishes setup after a model has been loaded.
    """
    self.__make_prior(
      self.model.get_final_shape(),
      train=self._get_parameter("learn_prior"))

    if self.__use_cuda:
      self.model.cuda()

  def __make_prior(self, shape, train):
    dim = np.prod(shape)

    self.model.prior_mu = nn.Parameter(
      torch.zeros((dim,)).float(), requires_grad=train)
    self.model.prior_logstd = nn.Parameter(
      torch.zeros((dim,)).float(), requires_grad=train)

  def __prior(self):
    return torch.distributions.Normal(
      self.model.prior_mu,
      torch.exp(self.model.prior_logstd))

  def __init_sampler(self):
    class DensitySampler:
      @staticmethod
      def sample(batch_size):
        with torch.no_grad():
          z = self._DensityIResNet__prior().sample((batch_size,))
          return self._DensityIResNet__model.inverse(z,
            max_iter=self._DensityIResNet__max_sample_iters)

    self.__sampler = utils.BufferedSampler(
      DensitySampler(),
      self._get_parameter("batch_size"))

  def __logpz(self, z):
    return self.__prior().log_prob(z).sum(dim=1)

  def logpx(self, inputs):
    self.ensure_available()
    return self.__logpx(inputs)

  def entropy(self, data, input_scaling=1.):
    """
    Returns the plug in estimate for continuous entropy of the given data.
    Data is assumed to return entries of the form (data, labels), but la
    bels
    will be ignored.

    Note that this does not take into account entropy change due to potential
    data transformations.

    Args:
      input_scaling: Scaling per dimension that has been applied to inputs
        compared to the data of which entropy is to be estimated.
    """
    self.ensure_available()

    batch_size = self._get_parameter("batch_size")

    entropyloader = torch.utils.data.DataLoader(data,
        batch_size=batch_size,
        num_workers=self.__loader_workers)

    entropy = 0.
    count = 0

    entropy_correction = -np.prod(data[0][0].shape)*np.log(input_scaling)

    progress_bar(0, len(entropyloader),
      pre_text=" Estimating ",
      post_text=" | Entropy: ------"
    )

    for batch_count, (inputs, _) in enumerate(entropyloader, 1):
      if self.__use_cuda:
        inputs = inputs.cuda()

      curr_entropy = -self.__logpx(inputs).mean()

      with torch.no_grad():
        entropy, count = utils.accumulate_mean(
            entropy, count, curr_entropy, len(inputs))

      progress_bar(batch_count, len(entropyloader),
        pre_text=" Estimating ",
        post_text=" | Entropy: {:6.3f}".format(entropy + entropy_correction)
      )

    return entropy.item() + entropy_correction

  def __logpx(self, inputs):
    """
    Returns the log probabilities of the given inputs.
    """
    result, trace = self.model(inputs)
    return self.__logpz(result) + trace

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

    init_batch = utils.get_batch_from_dataloader(
      self.__trainloader, self._get_parameter("init_batch"))

    if self.__use_cuda:
      init_batch = init_batch.cuda()

    with torch.no_grad():
      self.model(init_batch, ignore_logdet=True)

    self.model.train()

  def test(self, testset, metrics=["Loss"]):
    """
    Test the model on a testset by calculating the loss.

    Args:
      testset: List-like of (x,y) pairs for testing on.
      metrics: Useless parameter that is only provided as an interface for
        the TestResults reproducible. Must be ["Loss"].

    Return: List of the values of each metric in the same order as metrics
      input list.
    """
    assert metrics == ["Loss"]

    self.ensure_available()

    batch_size = self._get_parameter("batch_size")

    testloader = torch.utils.data.DataLoader(testset,
        batch_size=batch_size,
        num_workers=self.__loader_workers)

    loss_mean = 0.
    count = 0

    progress_bar(0, len(testloader),
      pre_text=" Testing ",
      post_text=" | Loss: ------"
    )

    for batch_count, (inputs, _) in enumerate(testloader, 1):
      if self.__use_cuda:
        inputs = inputs.cuda()

      curr_loss = utils_iResNet.bits_per_dim(self.logpx(inputs), inputs).mean()
      with torch.no_grad():
        loss_mean, count = utils.accumulate_mean(
            loss_mean, count, curr_loss, len(inputs))

      progress_bar(batch_count, len(testloader),
        pre_text=" Testing ",
        post_text=" | Loss: {:6.3f}".format(loss_mean)
      )

    return [loss_mean.item()]

  def _train(self):
    epoch = self._get_parameter("epochs")
    warmup_epochs = self._get_parameter("warmup_epochs")

    for batch_id, (inputs, _) in enumerate(self.__trainloader):
      cur_iters = (epoch - 1) * len(self.__trainloader) + batch_id

      if epoch <= warmup_epochs:
        this_lr = self._get_parameter("lr") * float(cur_iters) \
          / (warmup_epochs * len(self.__trainloader))
        utils_iResNet.update_lr(self.__optimizer, this_lr)
        del this_lr

      if self.__use_cuda:
        inputs = inputs.cuda()

      self.__optimizer.zero_grad()

      inputs = torch.autograd.Variable(inputs, requires_grad=True)

      loss = utils_iResNet.bits_per_dim(self.__logpx(inputs), inputs).mean()

      loss.backward()
      self.__optimizer.step()
      post_text = " | Loss: {:6.3f}".format(loss)

      progress_bar(batch_id + 1, len(self.__trainloader),
        pre_text=" Epoch {} ".format(epoch),
        post_text=post_text
      )

      del inputs, loss

  def _deinitialize_training(self):
    del self.__trainloader
    self.__eval()

  def __eval(self):
    self.__init_sampler()
    self.model.eval()

  @staticmethod
  def __make_representation_path(path):
    return join_paths(path, "_IResNet__state")

  def _unloadable(self):
    return True

  def unload(self):
    try:
      del self.model, self.__optimizer
    except AttributeError:
      pass
    
    super().unload()

  def _save(self, value, path):
    state = {
      'model': self.model,
      'optimizer': self.__optimizer.state_dict()
    }
    torch.save(state, self.__make_representation_path(path))
    super()._save(value, path)

  def _load(self, path):
    state = torch.load(self.__make_representation_path(path))
    self.model = state['model']
    self.__optimizer = self._get_optimizer(self.model.parameters())
    self.__optimizer.load_state_dict(state['optimizer'])
    self.__setup_model()
    if self._reproduction_depth() == 1:
      self.__eval()
    return super()._load(path)

  def __call__(self, inputs, *args, **kwargs):
    self.ensure_available()
    if self.__use_cuda:
      inputs = inputs.cuda()
    return self.model(inputs, *args, **kwargs)

  def sample(self, *args, **kwargs):
    """
    Sample from the training distribution.

    Args:
      max_iter: Maximum number of iterations for fixed point method used in
        model inverse.
    """
    self.ensure_available()
    return self.__sampler.sample(*args, **kwargs)
