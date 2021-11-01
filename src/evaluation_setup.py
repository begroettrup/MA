from interfaces.reproducible import Reproducible, ReproducibleFunction

from os.path import join as join_paths

import datasets

import algorithms.entropy_estimates as entrests
import algorithms.uniques as uniques

import interfaces.torch_utils as utils
import interfaces.losses as ilosses
from interfaces.image_io import save_grid
from interfaces.layer_hook import LayerHook
from interfaces.transformations import TupleUnpack, TransformConcat, ConcatOnSubset

import training_utils as tutls

from interfaces.entropy_estimation_utilities import \
  reproducible_entropy_estimate, make_ires_net, \
  make_entropic_data, dediscnorm_imgs, \
  dedisc_maybe_augment, train_val_sets, model_back_dens_split, \
  make_reproduction_resupnext, get_entropy_estimates, make_gradient_descent

from os import makedirs
import argparse

import torch
import numpy as np

# to enable communication of torch workers in spite of opening many files at once
torch.multiprocessing.set_sharing_strategy('file_system')

# constant things

parser = argparse.ArgumentParser(description="Evaluate entropy estimation methods "
  "on generated data.")

parser.add_argument('--reproducibles_base_dir',
  help='Base directory to save results under.')
parser.add_argument('--img_width', default=64,
  help='Width of generated images.')
parser.add_argument('--img_height', default=64,
  help='Height of generated images.')

parser.add_argument('--skip_discrete', action='store_true',
  help="Set to not compute discrete entropy estimates.")

parser.add_argument('--n_train', default=100000, type=int,
  help='Number of training samples.')
parser.add_argument('--n_test', default=10000, type=int,
  help='Number of test samples.')

parser.add_argument('--val_ratio', default=.1,
  help="Ratio of base data used for validation.")

parser.add_argument('--make_final_results', action='store_true',
  help="Set this flag to use the test set to create final results.")

parser.add_argument('--augment_data', action='store_true',
  help="Augment training data with random resize crop.")

parser.add_argument('--save_reconstructions',
  help='Save reconstructions in the given directory.')
args = parser.parse_args()

total_dims = args.img_height*args.img_width

# random number for np.random.uniform(0,1)
random_small = 0.1936778992815371
# random number for np.random.uniform(1,2)
random_large = 1.8219705708088378

warmup_epochs = 1
total_samples = args.n_train + args.n_test

reproduction_model_epochs = 1

norm_factor = 1/256.

if not args.skip_discrete:
  grassberger_without_collisions = ReproducibleFunction(
      entrests.entropy_grassberger_baseline,
      args=[total_samples],
      kwargs={"precision": 1e-6})

  print("Naive plug-in without collisions:",
    -np.log(1/total_samples))
  print("Grassberger estimate without collisions:",
      grassberger_without_collisions.reproduce_value())
  print("ANSB estimate without collisions: ∞")
  print()

def get_reproduction_models(forward_transformation, train_data, test_data, used_model):
  """
  Args:
    forward_transformation: Forward transformation.
    train_data: Training data set.
    used_model: Forward transformation network for unloading to free memory.
  
  Return: yields model, model_name, is_trained
  """
  yield make_reproduction_resupnext(
      train_data, test_data, forward_transformation,
      augment_data=args.augment_data
    )[0], "ResUpNeXt", True

  yield make_reproduction_resupnext(
      train_data, test_data, forward_transformation,
      augment_data=args.augment_data,
      loss = "L2"
    )[0], "ResUpNeXt.L2", True

  yield make_reproduction_resupnext(
      train_data, test_data, forward_transformation,
      augment_data=args.augment_data,
      loss = ConcatOnSubset(forward_transformation, ilosses.ReproducibleLoss("L1"), [0]),
      is_tt_loss = True
    )[0], "ResUpNeXt.TTL1", True

  yield make_gradient_descent(forward_transformation,
    (3,args.img_height,args.img_width)), "Gradient Descent", False

def entropy_evaluation(dataset, name, entropy, rundir):
  print("===",name,"===")
  print("Expected entropy:", entropy)
  # for factor in [random_small, random_large]:
  #   print("Expected continuous entropy with factor {}:".format(factor),
  #       entropy + np.log(factor)*total_dims)

  fixed_imgs_train = datasets.FixedDataSet(dataset, args.n_train)
  fixed_imgs_test = datasets.FixedDataSet(dataset, args.n_test)

  (imgs_model_train, imgs_model_val,
   imgs_back_train,  imgs_back_val,
   imgs_dens_train, imgs_dens_val) = model_back_dens_split(
    fixed_imgs_train, val_ratio=args.val_ratio)

  print("#data for original model training:", len(imgs_model_train),
        "; validation:", len(imgs_model_val))
  print("#data for backtransformation training:", len(imgs_back_train),
        "; validation:", len(imgs_back_val))
  print("#data for density training:", len(imgs_dens_train),
        "; validation:", len(imgs_dens_val))

  # density training set for gradient descent method which does not use training
  img_count = len(fixed_imgs_train)
  imgs_untrained_dens_train, imgs_untrained_dens_val = train_val_sets(
    fixed_imgs_train, img_count // 2, img_count, val_ratio=args.val_ratio)

  fixed_samples = utils.get_samples(
    dediscnorm_imgs(imgs_back_val, norm_factor=norm_factor), 16)
  fixed_samples_test = \
    utils.get_samples(
      dediscnorm_imgs(fixed_imgs_test, norm_factor=norm_factor), 16) \
    if args.make_final_results else None

  if args.save_reconstructions:
    makedirs(join_paths(args.save_reconstructions, rundir), exist_ok=True)
    save_grid(
      fixed_samples,
      join_paths(args.save_reconstructions, rundir,
        name.replace(" ", "_").lower() + ".png"
      )
    )

    if fixed_samples_test is not None:
      save_grid(
        fixed_samples_test,
        join_paths(args.save_reconstructions, rundir,
          "test_" + name.replace(" ", "_").lower() + ".png"
        )
      )

  if not args.skip_discrete:
    discrete_entropy_input = datasets.PreprocessedImages(dataset)
    discrete_entropy_input.set_parameter("output_length", total_samples)

    dataset_counts = ReproducibleFunction(
      uniques.get_counts,
      args=[discrete_entropy_input])

    grassberger = ReproducibleFunction(
        entrests.entropy_grassberger,
        args=[dataset_counts],
        kwargs={"precision": 1e-5})

    ansb = ReproducibleFunction(
        entrests.entropy_ansb,
        args=[dataset_counts])

    plugin = ReproducibleFunction(
        entrests.entropy_plug_in,
        args=[dataset_counts])

    print("Grassberger estimate:", grassberger.reproduce_value())
    print("ANSB estimate:", ansb.reproduce_value())
    print("Plug-in estimate:", plugin.reproduce_value())

  def make_ires_net_local(train_set, val_set):
    new_ires_net = make_ires_net(dedisc_maybe_augment(train_set,
      args.augment_data,
      # use ires_net scaling to make sure that entropy estimation works
      # this does not directly affect the results of entropy from
      # backtransformation in the ires_net
      scale=(.8,1.), norm_factor=norm_factor))
  
    return tutls.train_and_validate(new_ires_net,
      dediscnorm_imgs(val_set, norm_factor=norm_factor))[0]

  # iresnet trained on density data for comparison
  ires_net = make_ires_net_local(imgs_dens_train, imgs_dens_val)

  print("iResNet Entropy on base data:",
    reproducible_entropy_estimate(ires_net,
      dediscnorm_imgs(imgs_dens_val, norm_factor=norm_factor),
      norm_factor
  ).reproduce_value())

  # now set iresnet to the one to be inspected
  ires_net = make_ires_net_local(imgs_model_train, imgs_model_val)

  for layer in "1", "3", "5":
    print("---Layer", layer + "---")
    layer_transform = LayerHook(ires_net, ["stack", layer])

    forward_transformation = TransformConcat(layer_transform, TupleUnpack(0))

    with torch.no_grad():
      transformed_samples = forward_transformation(fixed_samples)

      if fixed_samples_test is not None:
        transformed_samples_test = forward_transformation(fixed_samples_test)

    for reproduction_model, reproduction_name, is_trained in get_reproduction_models(
      forward_transformation, imgs_back_train, imgs_back_val, ires_net):

      if is_trained:
        imgs_dens_train = imgs_dens_train
        imgs_dens_val = imgs_dens_val
      else:
        imgs_dens_train = imgs_untrained_dens_train
        imgs_dens_val = imgs_untrained_dens_val

      if args.save_reconstructions:
        save_grid(
          reproduction_model(transformed_samples),
          join_paths(args.save_reconstructions, rundir,
            (name + "-" + reproduction_name).replace(" ", "_").lower()
            + "." + layer + ".png"
          )
        )

        if fixed_samples_test is not None:
          save_grid(
            reproduction_model(transformed_samples_test),
            join_paths(args.save_reconstructions, rundir,
              "test_" + (name + "-" + reproduction_name).replace(" ", "_").lower()
              + "." + layer + ".png"
            )
          )

      def local_entropic_data(data):
        return make_entropic_data(data, forward_transformation, reproduction_model)

      entropic_train = local_entropic_data(imgs_dens_train)
      entropic_val = local_entropic_data(imgs_dens_val)
      entropic_test = local_entropic_data(fixed_imgs_test)

      entropic_train.ensure_available()
      entropic_val.ensure_available()
      if args.make_final_results:
        entropic_test.ensure_available()
      else:
        entropic_test = None

      reproduction_model.unload()
      ires_net.unload()

      results_val, results_test = get_entropy_estimates(
        entropic_train, entropic_val, entropic_test, norm_factor=norm_factor)

      print("Backtransformed entropy using " + reproduction_name + ":", results_val)
      if args.make_final_results:
        print("Backtransformed entropy using " + reproduction_name + " on test:",
          results_test)

  print()

if not args.reproducibles_base_dir:
  args.reproducibles_base_dir = Reproducible.get_base_directory()

# stochstical things that should be repeated over multiple runs
for reproducible_subdir in ["run1", "run2", "run3"]:
  print("Running in", reproducible_subdir)
  Reproducible.set_base_directory(
    join_paths(args.reproducibles_base_dir, reproducible_subdir))

  scene_images = datasets.SceneDataset("rpg_scene/scene.yml")
  assert(scene_images.fully_distinguishable())
  entropy_evaluation(scene_images, "RPG Scene Images",
    scene_images.max_entropy(), reproducible_subdir
  ) 
