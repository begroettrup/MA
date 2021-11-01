from interfaces.reproducible import ReproducibleFunction
from architectures.upsample_resnext import ReproducibleUpsampleResNeXt
from architectures.density_ires_net import DensityIResNet

from architectures.grad_desc_inverse import GradientDescentInverter, ZeroSuggestionModel

import interfaces.losses as ilosses

import torch

from datasets import Subset, SelfLabeledData, TransformedData, BacktransformedData, \
  PreprocessedImages

import numpy as np
import torchvision

import training_utils as tutls
import algorithms.uniques as uniqs

def estimate_entropy(model, inputs, dimwise_scaling):
  """
  Args:
    dimwise_scaling: Scaling applied on each transformation to get from data
      to inputs.
  """
  return model.entropy(inputs, dimwise_scaling)

def reproducible_entropy_estimate(model, inputs, dimwise_scaling=1.):
  """
  Args:
    dimwise_scaling: Scaling applied on each transformation to get from data
      to inputs.
  """
  return ReproducibleFunction(
    estimate_entropy,
    args=[model, inputs, dimwise_scaling],
    name="ReproducibleEntropyEstimate")

def make_ires_net(trainset, loader_workers=2,
  warmup_epochs=1, train_epochs=10):
  ires_net = DensityIResNet(use_cuda=True,
    name="Density Model", loader_workers=loader_workers)
  ires_net.set_parameter("input", trainset)
  ires_net.set_parameter("warmup_epochs", warmup_epochs)
  ires_net.set_parameter("epochs", train_epochs)
  ires_net.set_parameter("shuffle_data", False)
  ires_net.set_parameter("n_blocks", [2, 2, 2])
  ires_net.set_parameter("batch_size", 128)

  return ires_net

def ires_net_estimate(trainset, testset, factor=1., **kwargs):
  return reproducible_entropy_estimate(
    make_ires_net(trainset, **kwargs), testset, factor).reproduce_value()

def get_entropy_estimates(data_train, data_val, data_test=None,
  norm_factor=1/256.):
  data_train.ensure_available()

  iresnet = make_ires_net(data_train, loader_workers=2)

  iresnet, _ = tutls.train_and_validate(iresnet, data_val)

  estimate_val = reproducible_entropy_estimate(
    iresnet, data_val, dimwise_scaling=norm_factor).reproduction()

  if data_test is not None:
    estimate_test = reproducible_entropy_estimate(
      iresnet, data_test, dimwise_scaling=norm_factor).reproduction()
    return estimate_val, estimate_test
  else:
    return estimate_val, None

def get_counts_from_static(sdata):
  print("Calculating Static Counts")
  return uniqs.count_in_persistent(sdata.data())

def make_transformed_data(base_data, layer_transform):
  transformed_data = TransformedData()
  transformed_data.set_parameter("base_data", SelfLabeledData(base_data))
  transformed_data.set_parameter("transformation", layer_transform)
  transformed_data.set_parameter("transform_batch", True)

  return transformed_data

def make_entropic_data(base_data, layer_transform, reproduction_model,
  norm_factor=1/256.):
  entropic_data = BacktransformedData(batch_size=64, use_mmap=True)
  entropic_data.set_parameter("base_data", normalize_imgs(base_data))
  entropic_data.set_parameter(
    "forward_transformation", layer_transform)
  entropic_data.set_parameter(
    "backward_transformation", reproduction_model)
  # round results to integers before dediscretization
  entropic_data.set_parameter("discretize", True)
  entropic_data.set_parameter("dediscretization", "uniform")
  entropic_data.set_parameter("pixel_min", 0)
  entropic_data.set_parameter("pixel_max", 255)
  entropic_data.set_parameter("descaling_factor", 1 / norm_factor)

  return entropic_data

def normalize_imgs(data, norm_factor=1/256.):
  normalized_data = PreprocessedImages(data)
  normalized_data.set_parameter("post_factor", norm_factor)

  return normalized_data

def dediscnorm_imgs(data, norm_factor=1/256.):
  preprocessed_data = PreprocessedImages(data)
  preprocessed_data.set_parameter("post_factor", norm_factor)
  preprocessed_data.set_parameter("dediscretization", "uniform")

  return preprocessed_data

def augment_imgs(data,scale=(.25,1.),ratio=(.75,1.3333333333333333)):
  data = TransformedData(data,
    torchvision.transforms.RandomResizedCrop(
      (64,64),
      scale=scale,
      ratio=ratio
  ))
  data = TransformedData(data,
    torchvision.transforms.RandomHorizontalFlip()
  )
  return data

def dediscaugment(data,
                  norm_factor=1/256.,
                  scale=(.25,1.),
                  ratio=(.75,1.3333333333333333)):
  return augment_imgs(
    dediscnorm_imgs(data, norm_factor=norm_factor),
    scale=scale, ratio=ratio
  )

def dedisc_maybe_augment(data,
                         augment_data,
                         norm_factor=1/256.,
                         scale=(.25,1.),
                         ratio=(.75,1.3333333333333333)):
  data = dediscnorm_imgs(data, norm_factor=norm_factor)

  if augment_data:
    return augment_imgs(data, scale=scale, ratio=ratio)
  else:
    return data

def train_val_sets(dataset, start, stop, val_ratio):
  count = stop - start
  val_count = int(np.round(val_ratio * count))
  mid_stop = start + val_count
  return Subset(dataset, mid_stop, stop), Subset(dataset, start, mid_stop)

def model_back_dens_split(data, val_ratio):
  img_count = len(data)
  imgs_model_train, imgs_model_val = train_val_sets(
    data, 0, img_count // 2, val_ratio=val_ratio)
  imgs_back_train, imgs_back_val = train_val_sets(
    data, img_count // 2, 3 * img_count // 4, val_ratio=val_ratio)
  imgs_dens_train, imgs_dens_val = train_val_sets(
    data, 3 * img_count // 4, img_count, val_ratio=val_ratio)

  return (
    imgs_model_train, imgs_model_val,
    imgs_back_train, imgs_back_val,
    imgs_dens_train, imgs_dens_val
  )

def make_gradient_descent(forward_transformation, img_dims=(3,64,64),
  suggester=None):
  """
  Args:
    img_dims: (height, width) of images that should be produced
    suggester: Image suggestion model for initialization of gradient descent.
  """
  if suggester is None:
    suggester = ZeroSuggestionModel(img_dims)

  grad_desc_inverse = GradientDescentInverter(
    forward_transformation,
    suggester,
    use_cuda=True)

  # Sigmoid activation to be within value range of normalized image
  grad_desc_inverse.set_parameter("activation", torch.nn.Sigmoid())
  # stop if within 50 iterations loss hasn't improved by 3 percent
  grad_desc_inverse.set_parameter("early_stopping_params", (.97,50))

  return grad_desc_inverse

def make_reproduction_resupnext(data_train, data_val, layer_transform,
  loss = None, is_tt_loss = False, train_epochs=10, augment_data=True):
  """
  Args:
    data_train: Unnormalized, uncentered training data (i.e. range is [0, 255])
    data_val: Unnormalized, uncentered validation data (i.e. range is [0, 255])
  """
  def reannotate(base_data):
    if is_tt_loss:
      return SelfLabeledData(base_data)
    else:
      return base_data

  def local_transformed_data(base_data):
      return reannotate(make_transformed_data(base_data, layer_transform))

  if loss is None:
    loss = ilosses.FixedExponentLoss(.5, 2)

  reproduction_model = ReproducibleUpsampleResNeXt(
    use_cuda=True,
    out_shape=(3,64,64),
    # no workers since data is sampled through a model on GPU and transfering GPU data
    # through threads requires specific thread model
    loader_workers=0,
    name="Backtransformation Model")
  reproduction_model.set_parameter("input",
    local_transformed_data(dedisc_maybe_augment(data_train, augment_data)))
  reproduction_model.set_parameter("epochs", train_epochs)
  reproduction_model.set_parameter("block_depths", [2,2,2,2])
  reproduction_model.set_parameter("groups", 16)
  reproduction_model.set_parameter("width", 4)
  reproduction_model.set_parameter("batch_size", 64)
  reproduction_model.set_parameter("loss", loss)

  reproduction_model, test_results = tutls.train_and_validate(
    reproduction_model, local_transformed_data(dediscnorm_imgs(data_val)))
  reproduction_model.ensure_available()

  return reproduction_model, test_results