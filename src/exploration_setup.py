import interfaces.torch_utils as utils
import interfaces.losses as ilosses

from interfaces.torch_hub_model import TorchHubModel
from interfaces.layer_hook import LayerHook
from interfaces.image_io import save_grid
from interfaces.reproducible import Reproducible, ReproducibleFunction
from interfaces.test_results import ReproducibleModelTest, TestResults
from interfaces.transformations import ConcatOnSubset, Multiply, TransformChain
from interfaces.normalization import FixedValueModel

import interfaces.normalization as inormal

import argparse
from os.path import join as join_paths

import torch

import training_utils as tutls

from datasets import ImageNet64, UniqueDataset, DiffDataset, CombinedDataSet, SelfLabeledData

from interfaces.entropy_estimation_utilities import \
  reproducible_entropy_estimate, make_ires_net, \
  make_transformed_data, make_entropic_data, normalize_imgs, dediscnorm_imgs, \
  augment_imgs, dediscaugment, train_val_sets, model_back_dens_split, \
  make_reproduction_resupnext, get_entropy_estimates, make_gradient_descent

torch.multiprocessing.set_sharing_strategy('file_system')

norm_factor = 1/256.
img_height = 64
img_width = 64

# randomly chosen images from the validation set
# [ random.randrange(len(imgs_back_val)) for _ in range(16) ]
val_sample_ids  = [ 2400,  2259, 24572, 31150, 23663,  9758,  5052, 26727,
                   15521, 18018,  3224, 27629, 25062, 12815,  6182, 26203]

# randomly chosen images from the test set
# [ random.randrange(len(imgs_test)) for _ in range(16) ]
test_sample_ids = [31056, 41794, 32779, 40385, 33142, 25927, 36075,  9172,
                   26541, 18910, 40481, 30388, 36345, 24555, 20409,  6596]

parser = argparse.ArgumentParser(description="Estimate entropy on a model "
  "trained on ImageNet64 data.")

parser.add_argument('--reproducibles_dir', type=str, help='Base directory to save results under.')
parser.add_argument('--save_reconstructions',
  help='Save reconstructions in the given directory.')

parser.add_argument('--skip_entropy', action='store_true',
  help="""
  Set to not validate backtransformations. Intended for receiving samples without
  performing expensive computations.
  """)

parser.add_argument('--data',
  choices=['base', 'resnet', 'mobilenet', 'densenet'],
  nargs='+',
  default=[],
  help="""
    Data whose entropy to infer. For models the entropy of predetermined 
    layers is estimated unless the layers argument is set.
  """)

parser.add_argument('--layers', nargs='*',
  help="""
    Layers to compute entropy of. Layer paths should be separated with '/',
    e.g. "features/1" for the first feature layer of MobileNet.
    Ignored for base data.
  """)

parser.add_argument('--back_train_epochs', default=10, type=int,
  help="""
    Number of epochs to train the backpropagation model for.
  """)

# metric choices whose loss depends on the original data
def on_rounded(loss):
  return ConcatOnSubset(
    TransformChain(Multiply(1/norm_factor), torch.round, Multiply(norm_factor)),
    loss, [0])

metric_choices_back = {
  'L2Sqrt': lambda: ilosses.FixedExponentLoss(.5, 2),
  'L2': lambda: "L2",
  'L1': lambda: "L1",
  'RoundL2Sqrt': lambda: on_rounded(ilosses.FixedExponentLoss(.5, 2)),
  'RoundL2': lambda: on_rounded(ilosses.ReproducibleLoss("L2")),
  'RoundL1': lambda: on_rounded(ilosses.ReproducibleLoss("L1")),
}

# metric choices whose loss depends on the transformed data
metric_choices_tt = {
  'TTL1': lambda f: ConcatOnSubset(f, ilosses.ReproducibleLoss("L1"), [0]),
  'RoundTTL1': lambda f: on_rounded(ConcatOnSubset(f, ilosses.ReproducibleLoss("L1"), [0])),
}

parser.add_argument('--backtransformation', default='ResUpNeXt.L2Sqrt',
  choices=['ResUpNeXt.L2Sqrt', 'ResUpNeXt.L2', 'ResUpNeXt.TTL1', 'GradDesc',
           'GradDesc.ResUpNeXt'])

parser.add_argument('--val_metrics', nargs="+",
  choices=list(metric_choices_back.keys()) + list(metric_choices_tt.keys()),
  help="""
    Metrics that should be used for validation of the backtransformation.
  """)

parser.add_argument('--val_ratio', default=.1,
  help="Ratio of base data used for validation.")

parser.add_argument('--make_final_results', action='store_true',
  help="Whether to use the test set to create final results.")

args = parser.parse_args()

if args.reproducibles_dir:
  Reproducible.set_base_directory(args.reproducibles_dir)

imagenet64_dups = ImageNet64()
imagenet64_dups.set_parameter("normalized", False)
imagenet64_dups.set_parameter("centered", False)
imagenet64_dups_test = ImageNet64(parameters=imagenet64_dups.get_parameters())
imagenet64_dups_test.set_parameter("train", False)

imagenet64_test = UniqueDataset(imagenet64_dups_test)
imgs_test = imagenet64_test

imagenet64 = DiffDataset(base_data=imagenet64_dups, other=imagenet64_test)

combined_imagenet64 = CombinedDataSet(imagenet64, imagenet64_test)

(imgs_model_train, imgs_model_val,
 imgs_back_train,  imgs_back_val,
 imgs_dens_train,  imgs_dens_val) = model_back_dens_split(
  imagenet64, val_ratio=args.val_ratio)

print("#data for original model training:", len(imgs_model_train),
    "; validation:", len(imgs_model_val))
print("#data for backtransformation training:", len(imgs_back_train),
    "; validation:", len(imgs_back_val))
print("#data for density training:", len(imgs_dens_train),
    "; validation:", len(imgs_dens_val))
print("#data for testing:", len(imgs_test))

fixed_samples = utils.get_fixed_samples(normalize_imgs(imgs_back_val), val_sample_ids)
fixed_samples_test = utils.get_fixed_samples(normalize_imgs(imgs_test), test_sample_ids)

def data_mean(data):
  return ReproducibleFunction(inormal.data_mean, [data])

def save_unmodified_imgs(sample_data, start_string=""):
  save_grid(
    sample_data,
    join_paths(args.save_reconstructions,
      start_string + "samples.png"
    )
  )

if args.save_reconstructions:
  save_unmodified_imgs(fixed_samples)
  if args.make_final_results:
    save_unmodified_imgs(fixed_samples_test, "test_")

def print_entropy_estimates(data_name, data_train, data_val, data_test):
  print("=== {} ===".format(data_name))

  if not args.make_final_results:
    data_test = None

  results_val, results_test = get_entropy_estimates(
    data_train, data_val, data_test, norm_factor=norm_factor)
  
  print("Backtransformation IResNet entropy estimate on validation:", results_val)
  
  if args.make_final_results:
    print("Backtransformation IResNet entropy estimate on test:", results_test)

def validate_backtransformation(data_val, forward_transform, back_transform, set_name):
  val_data = make_transformed_data(normalize_imgs(data_val), forward_transform)
  val_data_forward_labels = SelfLabeledData()
  val_data_forward_labels.set_parameter("base_data", val_data)

  def fix_metric_choice_values(metric_choices, *args):
    new_metric_choices = {}

    for name, val in metric_choices.items():
      new_metric_choices[name] = val(*args)

    return new_metric_choices

  validation_results_back = ReproducibleModelTest(
    back_transform,
    val_data,
    fix_metric_choice_values(metric_choices_back),
    use_cuda=True,
    loader_workers=0
  )
  validation_results_tt = ReproducibleModelTest(
    back_transform,
    val_data_forward_labels,
    fix_metric_choice_values(metric_choices_tt, forward_transform),
    use_cuda=True,
    loader_workers=0
  )
  validation_results_back.ensure_available()
  validation_results_tt.ensure_available()

  back_transform.unload()
  forward_transform.unload()

  comparison_model = FixedValueModel(data_mean(normalize_imgs(data_val)))

  normalization_divisors_back = ReproducibleModelTest(
    comparison_model,
    # it doesn't matter which data is being used here as long as the labels fit
    # to not recompute every time constant data is used
    SelfLabeledData(normalize_imgs(data_val)),
    fix_metric_choice_values(metric_choices_back),
    use_cuda=True,
    loader_workers=0
  )
  normalization_divisors_tt = ReproducibleModelTest(
    comparison_model,
    val_data_forward_labels,
    fix_metric_choice_values(metric_choices_tt, forward_transform),
    use_cuda=True,
    loader_workers=0
  )

  # produce all losses at once to save computational effort
  numeric_results = {}
  norm_divisors = {}

  def append_results(validation_results, norm_divisor_results):
    for name, result, norm_div in zip(
        validation_results.metric_params,
        validation_results.reproduction(),
        norm_divisor_results.reproduction()
      ):
      numeric_results[name] = result
      norm_divisors[name] = norm_div

  if any(metric in metric_choices_back.keys() for metric in args.val_metrics):
    append_results(validation_results_back, normalization_divisors_back)
  if any(metric in metric_choices_tt.keys() for metric in args.val_metrics):
    append_results(validation_results_tt, normalization_divisors_tt)
  
  for name in args.val_metrics:
    print(set_name + " " + name + " loss:", numeric_results[name])
    print(" -- normalized:", numeric_results[name] / norm_divisors[name])

def estimate_entropy(model, model_name,
                     layer_transform, layer_str,
                     reproduction_model,
                     data_train, data_val, data_test):
  def local_entropic_data(data):
    return make_entropic_data(data, layer_transform, reproduction_model)

  entropic_train = local_entropic_data(data_train)
  entropic_val = local_entropic_data(data_val)
  entropic_test = local_entropic_data(data_test)

  # precompute transformed data, then unload reproduction model to free gpu memory
  entropic_train.ensure_available()
  entropic_val.ensure_available()
  if args.make_final_results:
    entropic_test.ensure_available()

  model.unload()
  reproduction_model.unload()

  print_entropy_estimates(model_name + "/" + layer_str,
    # augment after fixed transformation
    augment_imgs(entropic_train,
      # less heavy scaling to make sure the images have approximately the
      # same structure
      scale=(.8,1.)),
    entropic_val,
    entropic_test
  )

def make_backtransformation(data_train, data_val, layer_transform):
  # ['ResUpNeXt.L2Sqrt', 'ResUpNeXt.L2', 'ResUpNeXt.TTL1', 'GradDesc']
  if args.backtransformation == 'ResUpNeXt.L2Sqrt':
    return make_reproduction_resupnext(
      data_train, data_val, layer_transform,
      train_epochs=args.back_train_epochs)[0]
  elif args.backtransformation == 'ResUpNeXt.L2':
    return make_reproduction_resupnext(
      data_train, data_val, layer_transform, loss="L2",
      train_epochs=args.back_train_epochs)[0]
  elif args.backtransformation == 'ResUpNeXt.TTL1':
    return make_reproduction_resupnext(
      data_train, data_val, layer_transform,
      loss=ConcatOnSubset(layer_transform, ilosses.ReproducibleLoss("L1"), [0]),
      is_tt_loss = True,
      train_epochs=args.back_train_epochs
    )[0]
  elif args.backtransformation == 'GradDesc':
    return make_gradient_descent(layer_transform)
  elif args.backtransformation == 'GradDesc.ResUpNeXt':
    return make_gradient_descent(layer_transform,
      suggester=make_reproduction_resupnext(
        data_train, data_val, layer_transform,
        train_epochs=args.back_train_epochs)[0]
      )

def explore_layers_of(model, model_name,
  data_train_1, data_val_1, data_train_2, data_val_2,
  data_test, layer_names=None, additional_transformations={}):
  if layer_names is None:
    layer_names = model.model._modules.keys()

  for layer_name in layer_names:
    if type(layer_name) == str:
      layer_str = layer_name
    else:
      layer_str = ".".join(layer_name)

    print("---", model_name + "." + layer_str, "---")

    model.ensure_available()

    layer_transform = LayerHook(model, layer_name)

    if layer_str in additional_transformations:
      layer_transform = TransformChain(
        layer_transform, additional_transformations[layer_str])
      print("Using the configured additional transformation.")

    with torch.no_grad():
      samples = layer_transform(fixed_samples)
      samples_test = layer_transform(fixed_samples_test)

    model.unload()

    reproduction_model = make_backtransformation(
      data_train_1, data_val_1, layer_transform)

    if args.save_reconstructions:
      def save_imgs(sample_data, start_string=""):
        save_grid(
          reproduction_model(sample_data),
          join_paths(args.save_reconstructions,
            start_string + model_name.replace(" ", "_").lower() + "." + layer_str +
            "-" + args.backtransformation + ".png"
          )
        )

      save_imgs(samples)
      if args.make_final_results:
       save_imgs(samples_test, "test_")

    if args.val_metrics:
      validate_backtransformation(data_val_1, layer_transform, reproduction_model, "Back val")

      if args.make_final_results:
        validate_backtransformation(data_test, layer_transform, reproduction_model, "Back test")

    if not args.skip_entropy:
      estimate_entropy(model, model_name,
                       layer_transform, layer_str,
                       reproduction_model,
                       data_train_2, data_val_2, data_test)

def make_layers(default_layers):
  if args.layers:
    return list(map(lambda l: l.split('/'), args.layers))
  else:
    return default_layers

def test_model(name, model):
  if args.make_final_results:
    model_test_results = TestResults()
    model_test_results.set_parameter("model", model)
    model_test_results.set_parameter("testset", normalize_imgs(imgs_test))
    model_test_results.set_parameter("metrics", ["Loss", "Accuracy"])

    print(name + " test loss:", model_test_results.reproduction()[0])
    print(name + " test accuracy:", model_test_results.reproduction()[1])

if 'base' in args.data:
  print_entropy_estimates("original data",
    dediscaugment(imgs_dens_train),
    dediscnorm_imgs(imgs_dens_val),
    dediscnorm_imgs(imgs_test))

if 'resnet' in args.data:
  model = TorchHubModel("resnet50", use_cuda=True)
  model.set_parameter("input", dediscaugment(imgs_model_train))
  model.set_parameter("batch_size", 128)
  model.set_parameter("epochs", 30)
  model.set_parameter("optimizer_kwargs", {"momentum": .9})
  model.set_parameter("lr", 0.01)
  model.set_parameter("tracked_metrics", ["Loss", "Accuracy"])

  print("=== Training ResNet ===")
  model, model_test_results = tutls.train_and_validate(
    model, dediscnorm_imgs(imgs_model_val), add_metrics=["Accuracy"])
  print()
  test_model("ResNet", model)

  explore_layers_of(model, "ResNet",
    imgs_back_train, imgs_back_val, imgs_dens_train, imgs_dens_val,
    imgs_test,
    make_layers(
      # reversed map so deeper layers are treated first and potential problems
      # are thus uncovered sooner rather than later
      reversed([
        'relu', 'maxpool', 'layer1', 'layer2',
        'layer3', 'layer4', 'avgpool', 'fc',
      ])
    )
  )

if 'mobilenet' in args.data:
  model = TorchHubModel("mobilenet_v2", use_cuda=True)
  model.set_parameter("input", dediscaugment(imgs_model_train))
  model.set_parameter("batch_size", 128)
  model.set_parameter("epochs", 30)
  model.set_parameter("optimizer_kwargs", {"momentum": .9})
  model.set_parameter("lr", 0.01)
  model.set_parameter("tracked_metrics", ["Loss", "Accuracy"])

  print("=== Training MobileNet ===")
  model, model_test_results = tutls.train_and_validate(
    model, dediscnorm_imgs(imgs_model_val), add_metrics=["Accuracy"])
  print()
  test_model("MobileNet", model)

  explore_layers_of(model, "MobileNet",
    imgs_back_train, imgs_back_val, imgs_dens_train, imgs_dens_val,
    imgs_test,
    make_layers(
      # reversed map so deeper layers are treated first and potential problems
      # are thus uncovered sooner rather than later
      reversed([ ['features', layer]
          for layer in  ['0', '1', '3', '6', '13', '17', '18']]
        + ['classifier'])
    ),
    # this layer needs to be manually done since it is implied in
    # functional form
    { 'features.18': torch.nn.AdaptiveAvgPool2d((1,1)) }
  )

if 'densenet' in args.data:
  model = TorchHubModel("densenet161", use_cuda=True)
  model.set_parameter("input", dediscaugment(imgs_model_train))
  model.set_parameter("batch_size", 128)
  model.set_parameter("epochs", 30)
  model.set_parameter("optimizer_kwargs", {"momentum": .9})
  model.set_parameter("lr", 0.01)
  model.set_parameter("tracked_metrics", ["Loss", "Accuracy"])

  print("=== Training DenseNet ===")
  model, model_test_results = tutls.train_and_validate(
    model, dediscnorm_imgs(imgs_model_val), add_metrics=["Accuracy"])
  print()
  test_model("DenseNet", model)

  explore_layers_of(model, "DenseNet",
    imgs_back_train, imgs_back_val, imgs_dens_train, imgs_dens_val,
    imgs_test,
    make_layers(
      # reversed map so deeper layers are treated first and potential problems
      # are thus uncovered sooner rather than later
      reversed([ ['features', layer]
          for layer in  ['relu0', 'pool0', 'denseblock1', 'denseblock2',
          'denseblock3', 'denseblock4', 'norm5']]
        + ['classifier']
      )
    ),
    {'features.norm5': TransformChain(
      torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((1,1)))}
  )
