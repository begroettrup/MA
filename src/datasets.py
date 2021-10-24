import torch
import torchvision as tv

from interfaces.iterables import get_from_multiple

import itertools

from pathlib import Path

import numpy as np

import yaml

import inspect

import pickle

import interfaces.scene_generator as scene_generator

import os
from os.path import join as join_paths

from tempfile import TemporaryDirectory

from functools import partial

import interfaces.torch_utils as utils
from interfaces.torch_utils import to_batch, from_batch, batch_count, BufferedSampler
from interfaces.command_line import progress_bar

from interfaces.reproducible import VirtualReproducible, Reproducible

import algorithms.numpy_utils as npu
import algorithms.uniques as uniqs

def is_static_set(data):
  try:
    return data.is_static()
  except AttributeError:
    return False

def check_inrange(self, index, my_name="dataset"):
  """
  Checks whether the index is in range of own length and raises an appropriate
  exception if it isn't.

  Args:
    self: Object to check the length of.
    index: Index which might be out of range.
    my_name: String to refer to the object by in the error message.
  """
  if index >= len(self):
    raise IndexError(my_name + " index out of range")

class ImageNet64(VirtualReproducible):
  def __init__(self, data_path="ImageNet64", parameters={}, **kwargs):
    """
    Get a new ImageNet64 data set.

    Args:
    data_path: Path to the directory where the image net dataset is stored.
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "train": Whether to load training data. Loads testing data if False.
        - "normalized": If true, values are normalized such that the maximum pixel
          value is mapped to 1. before subtracting the mean.
        - "dediscretization": None or "uniform". Uniform dediscretization adds
          a uniform distribution per pixel with width equal to the distance
          between two discrete values.
        - "centered": Centers the data around the data mean by subtracting it.
        - "uniques": Drop non-unique elements.
    """
    parameters = {
      **{
        "_ImageNet64_version": "1.0.2",
        "train": True,
        "normalized": True,
        "dediscretization": None,
        "centered": True
      },
      **parameters}

    self.__data_path = Path(data_path)

    super().__init__(parameters=parameters, **kwargs)

  def __make_file_path(self, file_name):
    return join_paths(self.__data_path, "." + file_name)

  def __make_meta_path(self):
    return self.__make_file_path("meta.pkl")

  def data(self):
    """
    Access to the data array.
    """
    self.ensure_available()
    return self.__data

  def transform(self, x):
    """
    Transform a single data item.
    """
    x = torch.from_numpy(x.copy())
    for f in self.__transforms:
      x = f(x)

    return x

  def labels(self):
    """
    Access to the labels list.
    """
    self.ensure_available()
    return self.__labels

  def _produce(self):
    path_data_train = self.__make_file_path("data_train.npy")
    path_labels_train = self.__make_file_path("labels_train.npy")
    path_data_val = self.__make_file_path("data_val.npy")
    path_labels_val = self.__make_file_path("labels_val.npy")

    is_train = self._get_parameter("train")

    try:
      # First check for efficient representation
      with open(self.__make_meta_path(), "rb") as f:
        meta_info = pickle.load(f)

      if is_train:
        self.__labels = np.load(path_labels_train)
      else:
        self.__labels = np.load(path_labels_val)
    except FileNotFoundError:
      # if files doesn't exist yet, create them
      print("First time loading ImageNet64 data. This may take a while.")

      meta_info = {}

      train_list = list(map(lambda x: "train_data_batch_{}".format(x+1), range(10)))

      for fn in train_list + ["val_data"]:
        fp = self.__data_path / fn
        if not fp.exists():
          raise FileNotFoundError(str(fp) + " does not exist. Please make "
            "sure that ImageNet64 data has been downloaded and extracted "
            "into the specfied directory.")

      meta_info["nval"] = 50000
      meta_info["ntrain"] = 1281167
      meta_info["shape"] = (3,64,64)

      data_val = np.memmap(path_data_val,
        dtype="uint8", mode="w+",
        shape=(meta_info["nval"],) + meta_info["shape"])
      labels_val = np.empty(meta_info["nval"], dtype="int64")

      with (self.__data_path / "val_data").open("rb") as f:
        val_data = pickle.load(f)

        new_shape = (len(val_data["data"]), 3, 64, 64)
        
        data_val[:] = val_data["data"].reshape(new_shape)
        labels_val = val_data["labels"]

        del val_data

      data_train = np.memmap(path_data_train,
        dtype="uint8", mode="w+",
        shape=(meta_info["ntrain"],) + meta_info["shape"])
      labels_train = np.empty(meta_info["ntrain"], dtype="int64")

      start = 0

      for file_name in train_list:
        with (self.__data_path / file_name).open("rb") as f:
          new_data = pickle.load(f)

          if "mean" not in meta_info:
            meta_info["mean"] = new_data["mean"]

          end = start + len(new_data["data"])
          new_shape = (len(new_data["data"]), 3, 64, 64)

          data_train[start:end] = new_data["data"].reshape(new_shape)
          labels_train[start:end] = new_data["labels"]

          start = end
          del new_data

      # labels would start at 1 without change
      labels_train = np.array(labels_train) - 1
      labels_val = np.array(labels_val) - 1

      # save data
      data_train.flush()
      data_val.flush()

      np.save(path_labels_train, labels_train)
      np.save(path_labels_val, labels_val)

      with open(self.__make_meta_path(), "wb") as f:
        pickle.dump(meta_info, f)

      if is_train:
        self.__labels = labels_train
      else:
        self.__labels = labels_val

      del data_train, data_val, labels_train, labels_val

    if is_train:
      self.__len = meta_info["ntrain"]
      self.__data = np.memmap(path_data_train,
        dtype="uint8", mode="r",
        shape=(meta_info["ntrain"],) + meta_info["shape"])
    else:
      self.__len = meta_info["nval"]
      self.__data = np.memmap(path_data_val,
        dtype="uint8", mode="r",
        shape=(meta_info["nval"],) + meta_info["shape"])

    self.__transforms = []

    datamean = torch.tensor(meta_info["mean"]).type(torch.float).view(3,64,64)

    total_shift = 0.

    if self._get_parameter("dediscretization") == "uniform":
      self.__transforms.append(lambda x: x + torch.empty_like(x).uniform_(0., 1.))
      total_shift += 1
    elif self._get_parameter("dediscretization") is not None:
      raise ValueError("Unknown dediscretization method '"
        + self._get_parameter("dediscretization") + "'")

    if self._get_parameter("centered"):
      # bind current value of total_shift to default
      self.__transforms.append(lambda x, datamean=datamean, total_shift=total_shift:
        x - datamean - total_shift / 2)

    if self._get_parameter("normalized"):
      self.__transforms.append(lambda x, total_shift=total_shift: x / (255. + total_shift))

  def __len__(self):
    self.ensure_available()
    return self.__len

  def __getitem__(self, idx):
    self.ensure_available()

    x = self.transform(self.__data[idx])

    y = self.__labels[idx]

    return x, y - 1

  def is_static(self):
    """
    Returns whether this dataset is static, i.e. doesn't change.
    """
    return self._get_parameter("dediscretization") is None

class MNIST(VirtualReproducible):
  def __init__(self, parameters={}, **kwargs):
    """
    Get a new MNIST data set.

    Args:
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "train": Whether to load training data. Loads testing data if False.
        - "normalized": If true, values are normalized such that pixel value
          256. is mapped to 1.
        - "dediscretization": None or "uniform". Uniform dediscretization adds
          a uniform distribution per pixel with width equal to the distance
          between two discrete values.
        - "zero_centered": Centers the data around zero, i.e. data will be in
          the range [-0.5,0.5] resp [-128,128) if set.
        - "padding": Add zero padding to receive 32x32 instead of 28x28 images.
    """
    parameters = {
      **{
        "_MNIST_version": "2.0.0",
        "train": True,
        "normalized": True,
        "dediscretization": None,
        "zero_centered": False,
        "padding": False,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    transforms = [tv.transforms.ToTensor(), lambda x: x*255.]

    if self._get_parameter("padding"):
      transforms.append(partial(torch.nn.functional.pad, pad=[2,2,2,2]))

    if self._get_parameter("dediscretization") == "uniform":
      transforms.append(lambda x: x + torch.empty_like(x).uniform_(0., 1.))
    elif self._get_parameter("dediscretization") is not None:
      raise ValueError("Unknown dediscretization method '"
        + self._get_parameter("dediscretization") + "'")

    if self._get_parameter("zero_centered"):
      transforms.append(lambda x: x - 128)

    if self._get_parameter("normalized"):
      transforms.append(lambda x: x / 256.)

    self.__dataset = tv.datasets.MNIST("~/.pytorch/datasets",
      train=self._get_parameter("train"), download=True,
      transform=tv.transforms.Compose(transforms))

  def __len__(self):
    self.ensure_available()
    return len(self.__dataset)

  def __getitem__(self, idx):
    self.ensure_available()
    return self.__dataset[idx]

  def is_static(self):
    """
    Returns whether this dataset is static, i.e. doesn't change.
    """
    return self._get_parameter("dediscretization") is None

class MMapDataSet(Reproducible):
  """
  A dataset whose data is created within persistent memory and saved by moving
  the respective file.

  Using classes should implement _make_data_in. Data can afterwards be accessed
  via data() and labels can be accessed via labels()
  """
  def __init__(self, parameters={}, **kwargs):
    parameters = {
      **{
        "_MMapDataSet__version": "1.0.1"
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def __getitem__(self, idx):
    self.ensure_available()
    return torch.from_numpy(np.array(self.__data[idx])), self.__labels[idx]

  def __len__(self):
    self.ensure_available()
    return len(self.__data)

  def _make_data_in(self, filename):
    """
    Create the data within the file given by filename. Should return a tuple
    (data, labels). data should be a memmap array.
    """
    raise NotImplementedError()

  def data(self):
    """
    Access to the data array.
    """
    self.ensure_available()
    return self.__data

  def labels(self):
    """
    Access to the labels list.
    """
    self.ensure_available()
    return self.__labels

  def _produce(self):
    self.__tmpdir = TemporaryDirectory()
    self.__data, self.__labels = self._make_data_in(
      join_paths(self.__tmpdir.name, "data"))
    self.__data.flush()

  def _save(self, value, path):
    # make metadata
    meta_data = {
      "data_shape": self.__data.shape,
      "data_dtype": self.__data.dtype
    }

    # make temporary file permanent
    try:
      os.rename(
        join_paths(self.__tmpdir.name, "data"),
        self.__make_data_path(path))
    except OSError:
      # can't move between devices
      data = npu.mmap_copy(np.memmap(
        join_paths(self.__tmpdir.name, "data"),
        mode="r",
        shape=meta_data["data_shape"],
        dtype=meta_data["data_dtype"]
      ), filename=self.__make_data_path(path))
      data.flush()
      del data
    # clean up tmpdir
    del self.__tmpdir

    # access created data array
    self.__data = self.__make_data_memmap("r", path, meta_data)
    with open(self.__make_labels_path(path), "wb") as file:
      pickle.dump(self.__labels, file)
    with open(self.__make_meta_path(path), "wb") as file:
      pickle.dump(meta_data, file)

    return super()._save(value, path)

  def _unloadable(self):
    return True

  def _load(self, path):
    with open(self.__make_meta_path(path), "rb") as file:
      meta_data = pickle.load(file)
    with open(self.__make_labels_path(path), "rb") as file:
      self.__labels = pickle.load(file)
    self.__data = self.__make_data_memmap("r", path, meta_data)

    return super()._load(path)

  def unload(self):
    try:
      del self.__data, self.__labels, self.__meta_data
    except AttributeError:
      pass
    super().unload()

  @staticmethod
  def __make_data_path(path):
    return join_paths(path, "_MMapDataSet__data")

  @staticmethod
  def __make_labels_path(path):
    return join_paths(path, "_MMapDataSet__labels")

  @staticmethod
  def __make_meta_path(path):
    return join_paths(path, "_MMapDataSet__meta")

  @staticmethod
  def __make_data_memmap(mode, path, meta_data):
    return np.memmap(
      MMapDataSet.__make_data_path(path),
      mode=mode,
      shape=meta_data["data_shape"],
      dtype=meta_data["data_dtype"]
    )

class SubsetOfBaseData(Reproducible):
  def __init__(self, base_data=None, parameters={}, **kwargs):
    """
    A dataset which is a subset of some base data. Subclasses should implement
    data() and labels() methods or have them implemented in a superclass.
    If a subclass implements _produce it should also call _init_base_set() within
    that method.
    """
    parameters = {
      **{
        "_SubsetOfBaseData__version": "1.0.0",
        "base_data": base_data,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _init_base_set(self):
    base_data = self._get_parameter("base_data")

    try:
      self.__transform = base_data.transform
    except AttributeError:
      self.__transform = torch.from_numpy

  def transform(self, x):
    return self.__transform(x)

  def __getitem__(self, idx):
    self.ensure_available()

    x = self.data()[idx]
    x = self.__transform(x)

    return x, self.labels()[idx]

  def _produce(self):
    self._init_base_set()
    super()._produce()

  def _load(self, path):
    self._init_base_set()
    return super()._load(path)

class DiffDataset(SubsetOfBaseData, MMapDataSet):
  """
  Removes all elements in the dataset that are also members of other as well
  as all duplicates.
  """
  def __init__(self, base_data=None, other=None,
    *datasets, parameters={}, **kwargs):

    parameters = {
      **{
        "_DiffDataset__version": "1.0.3",
        "other": other
      },
      **parameters}

    super().__init__(base_data=base_data,parameters=parameters, **kwargs)

  def _make_data_in(self, filename):
    print("Subtracting datasets. This may take a while.")
    base = self._get_parameter("base_data")
    other = self._get_parameter("other")

    max_size = max(len(base), len(other))
    index_size = npu.bytes_for_size(max_size)

    def dataset_arrays(dataset, marker):
      return [
          npu.to_2d(dataset.data()),
          # marker comes first so all elements of the same dataset are in one block
          npu.to_2d(np.full(len(dataset), marker)),
          npu.index_array(len(dataset), index_size)
        ]

    base_marker = 1
    other_marker = 0

    print("-> Preparing working data.")
    flat_combined = npu.combine_data(
      [
        dataset_arrays(base, base_marker),
        # other elements should always be first
        dataset_arrays(other, other_marker)
      ],
      # add labels and markers into data behind each element (axis=-1)
      # then add base and other data after each other (axis=0)
      axis=(-1,0),
      empty_constructor=npu.tmp_mmap
    )

    # sort combined in place
    print("-> Sorting data.")
    npu.lexsort_in_place(flat_combined)

    print("-> Finding unique elements.")
    interesting = np.logical_and(
      # only consider the first of each element
      # if both datasets contain an element, this will be an element of other
      npu.changes(flat_combined[:,:-index_size-1]),
      # only retain elements from the base set
      flat_combined[:,-index_size-1] == base_marker
    )

    print("-> Creating final data representation.")
    indices = npu.collapse_index_array(flat_combined[interesting,-index_size:])
    # sort indices to not change data order
    indices.sort()

    data = npu.mmap_copy(base.data()[indices], filename)

    return data, np.array(base.labels())[indices]

  def is_static(self):
    return True

class CombinedDataSet(MMapDataSet):
  def __init__(self, *datasets, parameters={}, **kwargs):
    parameters = {
      **{
        "_CombinedDataSet__version": "1.1.0",
        "_CombinedDataSet__ndata": len(datasets)
      },
      **parameters}

    for i in range(len(datasets)):
      parameters[i] = datasets[i]

    super().__init__(parameters=parameters, **kwargs)

  def __get_count(self):
    return self._get_parameter("_CombinedDataSet__ndata")

  def _make_data_in(self, filename):
    datas = []
    labelss = []

    for i in range(self.__get_count()):
      ds = self._get_parameter(i)
      datas.append(ds.data())
      labelss.append(ds.labels())

    data = npu.combine_data(datas, npu.mmap_constructor_on(filename), axis=0)
    labels = npu.combine_data(labelss, axis=0)

    return data, labels

class FixedDataSet(MMapDataSet):
  """
  Fixes a random data set in place for reuse.
  """
  def __init__(self, base_data=None, size=None, parameters={}, **kwargs):
    if size is None and base_data is not None:
      size = len(base_data)

    parameters = {
      **{
        "_FixedDataSet__version": "1.0.1",
        "base_data": base_data,
        "size": size
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _make_data_in(self, filename):
    base_data = self._get_parameter("base_data")
    size = self._get_parameter("size")

    progress_bar(0, size,
      pre_text=" Computing Dataset Fixture ")

    for i in range(size):
      d, l = base_data[i % len(base_data)]
      try:
        data[i,:] = d
      except UnboundLocalError:
        data = np.memmap(filename,
          mode='w+',
          shape=(size,) + d.shape,
          dtype=d.numpy().dtype
        )
        labels = []
        data[i,:] = d
      labels.append(l)
      progress_bar(i+1, size,
        pre_text=" Computing Dataset Fixture ")

    return data, labels

  def is_static(self):
    return True

class StaticTransformedData(MMapDataSet):
  def __init__(self, batch_size=32, parameters={}, **kwargs):
    """
    Transforms a dataset and saves the transformed data.

    Args:
      batch_size: Number of samples that will be passed to the transformation
        function as a batch. If None, all samples will be passed at once.
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "base_data": Original dataset that is to be transformed.
        - "transformation": Transformation to apply to the data.
        - "transform_batch": If set the transformation function will receive
          single element batches and its return will be treated as single
          element batch and transformed into non-batched representation.
    """
    parameters = {
      **{
        "_StaticTransformedData__version": "1.0.0",
        "base_data": None,
        "transformation": None,
        "transform_batch": False
      },
      **parameters}

    self.__batch_size = batch_size

    super().__init__(parameters=parameters, **kwargs)

  def _make_data_in(self, filename):
    data = None

    base_data = self._get_parameter("base_data")
    transform = self._get_parameter("transformation")

    data_length = len(base_data)
    labels = []

    if len(base_data) > 0 and self.__batch_size > 1:
      input_shape = base_data[0][0].shape

    for start in range(0, data_length, self.__batch_size):
      end = min(start + self.__batch_size, data_length)

      if self.__batch_size == 1:
        inputs = to_batch(base_data[start][0])
      else:
        inputs = torch.empty((end - start,) + input_shape)
        for i in range(end - start):
          inputs[i], label = base_data[start + i]
          labels.append(label)

      with torch.no_grad():
        outputs = transform(inputs)
      outputs = outputs.cpu().numpy()

      if data is None:
        data = np.memmap(
          filename,
          dtype=outputs.dtype,
          mode='w+',
          shape=(data_length,) + outputs.shape[1:]
        )

      data[start:end] = outputs

      del outputs, inputs

      progress_bar(start // self.__batch_size + 1,
        batch_count(len(base_data), self.__batch_size),
        pre_text=" Computing Dataset ")

    self._unload_parameters()

    return data, labels

class Subset(SubsetOfBaseData, VirtualReproducible):
  """
  Returns a version of the dataset with only a subset of elements included.
  """
  def __init__(self, base_data=None, limit1=None, limit2=None, step=1,
    parameters={}, **kwargs):
    """
    Args:
      base_data: Dataset to produce a subset of.
      limit1: Stop if limit2 is not set, start otherwise.
      limit2: If set, limit1 will be start and this will be stop.
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "base_data": Original dataset that is to be reduced. Must have a
          data() method which receives all of the data as numpy array and a
          labels() method to receive all labels as numpy array.
        - "start": Start index of the subset.
        - "stop": After end index of the subset.
        - "step": Step by which elements are addressed.
    """
    if limit2 is not None:
      start = limit1
      stop = limit2
    else:
      start = None
      stop = limit1

    parameters = {
      **{
        "_Subset__version": "1.0.1",
        "start": start,
        "stop": stop,
        "step": step
      },
      **parameters}

    super().__init__(base_data=base_data,parameters=parameters, **kwargs)

  def data(self):
    return self.__data.data()[self.__start:self.__stop:self.__step]

  def labels(self):
    return np.array(self.__data.labels())[self.__start:self.__stop:self.__step]

  def _produce(self):
    self._init_base_set()
    self.__start = self._get_parameter("start")
    self.__stop = self._get_parameter("stop")
    self.__step = self._get_parameter("step")
    self.__data = self._get_parameter("base_data")

    if self.__start is None:
      self.__start = 0

    if self.__stop is None:
      self.__stop = len(self.__data)

    def get_absolute_bound(bound):
      if bound < 0:
        return max(0, bound + len(self.__data))
      else:
        return bound

    self.__stop = get_absolute_bound(self.__stop)
    self.__stop = min(self.__stop, len(self.__data))

    self.__start = get_absolute_bound(self.__start)
    self.__start = min(self.__start, self.__stop)

  def __len__(self):
    self.ensure_available()
    return (self.__stop - self.__start) // self.__step

  def __getitem__(self, idx):
    self.ensure_available()
    idx = idx*self.__step + self.__start
    if idx < self.__stop:
      return self.__data[idx]
    else:
      raise IndexError("Index " + str(idx) + " is out of bounds "
        "for subset of length " + str(self.__len__()) + ".")

  def is_static(self):
    return is_static_set(self._get_parameter("base_data"))

class UniqueDataset(SubsetOfBaseData, MMapDataSet):
  """
  Returns a version of the dataset with all exact duplicates removed.
  """
  def __init__(self, base_data=None, parameters={}, **kwargs):
    """
    Transforms a dataset and saves the transformed data.

    Args:
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "base_data": Original dataset that is to be reduced. Must have a
          data() method which receives all of the data as numpy array and a
          labels() method to receive all labels as numpy array.
    """
    parameters = {
      **{
        "_UniqueDataset__version": "1.0.2",
      },
      **parameters}

    super().__init__(base_data=base_data, parameters=parameters, **kwargs)

  def _make_data_in(self, filename):
    print("Removing duplicates in dataset. This may take a while.")
    data = self._get_parameter("base_data").data()
    labels = self._get_parameter("base_data").labels()
    labels = np.array(labels)

    sorted_data, indices = npu.mmap_argsort(data)

    indices = np.array(indices[npu.changes(sorted_data)], dtype=np.long)
    # sort indices to not change data order
    indices.sort()

    data = npu.mmap_copy(data[indices], filename)

    return data, labels[indices]

  def is_static(self):
    return True

class TransformedData(VirtualReproducible):
  def __init__(self, base_data=None, transformation=None, transform_batch=False,
    parameters={}, **kwargs):
    """
    Get a new transformed data set.

    Args:
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "base_data": Original dataset that is to be transformed.
        - "transformation": Transformation to apply to the data.
        - "transform_batch": If set the transformation function will receive
          single element batches and its return will be treated as single
          element batch and transformed into non-batched representation.
    """
    parameters = {
      **{
        "_TransformedData__version": "2.0.0",
        "base_data": base_data,
        "transformation": transformation,
        "transform_batch": transform_batch
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def __len__(self):
    self.ensure_available()
    return len(self._get_parameter("base_data"))

  def _produce(self):
    t = self._get_parameter("transformation")

    if self._get_parameter("transform_batch"):
      t = partial(lambda f, x: from_batch(f(to_batch(x))), t)

    self.__transform = t
    self.__data = self._get_parameter("base_data")

  def __getitem__(self, idx):
    self.ensure_available()
    val, label = self.__data[idx]

    with torch.no_grad():
      out = self.__transform(val)
      out.requires_trad = False
      return (out, label)

  def _unloadable(self):
    return True

  def unload(self):
    try:
      del self.__transform, self.__data
    except AttributeError:
      pass
    super().unload()

  def is_static(self):
    return is_static_set(self._get_parameter("base_data"))

class SelfLabeledData(VirtualReproducible):
  """
  A data set but transformed such that its data is equal to its labels.
  """
  def __init__(self, base_data=None, parameters={}, **kwargs):
    """
    Get a new self labeled data set.

    Args:
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "base_data": Original dataset that is to be relabeled. Prece
    """
    parameters = {
      **{
        "_SelfLabeledData__version": "1.0.0",
        "base_data": base_data,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def __len__(self):
    self.ensure_available()
    return len(self._get_parameter("base_data"))

  def _produce(self):
    self.__data = self._get_parameter("base_data")

  def __getitem__(self, idx):
    self.ensure_available()
    val, _ = self.__data[idx]
    return (val, val)

  def is_static(self):
    return is_static_set(self._get_parameter("base_data"))

class BacktransformedData(Reproducible):
  """
  An image dataset that is transformed by some function and then transformed
  back into data that has the same structure.
  """
  def __init__(self, batch_size=32, use_mmap=False, parameters={}, **kwargs):
    """
    Get a new self labeled data set.

    Args:
      batch_size: Number of samples that will be passed to the transformation
        function as a batch. If None, all samples will be passed at once.
      parameters: Dictionary of parameters to set. Default parameters that
        change the dataset behavior are:
        - "base_data": Original dataset that will be transformed and back
          again.
        - "forward_transformation": Function that transforms the data into a
          different structure.
        - "backward_transformation": Function that transforms the data back to
          its original structure.
        - "dediscretization": None or "uniform". Uniform dediscretization adds
          a uniform distribution on [0,1] per pixel.
        - "pixel_min": Minimum all indiviudal pixel values will be raised to if
          below.
        - "pixel_max": Maximum all indiviudal pixel values will be lowered to
          if above.
        - "descaling_factor": The factor the data would need to be multiplied
          with for denormalization. This needs to be set even if denormalization
          is not desired to correctly apply clamping and dediscretization.
        - "denormalize": Set if the data should be returned in denormalized
          form. If not set, the data scale will not be modified.
        - "discretize": True or False. If set, data will be rounded to the
          nearest integer after descaling. This will be applied before any
          desdiscretization.
        - "retain_shape": If set outputs which are smaller than the initial
          input will be zero padded symmetrically as 1-dimensional tensors
          (excluding batch dimension) and outputs of different shape and same
          total size (including those smaller before padding) than their
          respective inputs will be reshaped to match.
          If the outputs would be bigger than their inputs, an error will be
          thrown.
    """
    parameters = {
      **{
        "_BacktransformedData__version": "1.3.0",
        "base_data": None,
        "forward_transformation": None,
        "backward_transformation": None,
        "discretize": False,
        "dediscretization": None,
        "denormalize": False,
        "descaling_factor": 1.,
        "pixel_min": None,
        "pixel_max": None,
        "retain_shape": True,
      },
      **parameters}

    self.__batch_size = batch_size
    self.__mmap = use_mmap

    super().__init__(parameters=parameters, **kwargs)

  @staticmethod
  def __make_data_path(path):
    return join_paths(path, "_BacktansformedData__data")

  @staticmethod
  def __make_mmap_data_path(path):
    return join_paths(path, "_BacktansformedData__mmap_data")

  @staticmethod
  def __make_meta_path(path):
    return join_paths(path, "_BacktansformedData__meta_data")

  def __save_data(self, path):
    if self.__data is not None:
      torch.save(self.__data, self.__make_data_path(path))

  def _save(self, value, path):
    if is_static_set(self._get_parameter("base_data")):
      # calculate and save data set
      if self.__mmap:
        self.__data = self.__make_transformed_data(path)
      else:
        self.__data = self.__make_transformed_data()
        self.__save_data(path)
    else:
      self.__data = None
      self.__save_data(path)

    return super()._save(value, path)

  def __initialize(self):
    """
    Called when the object is produced or loaded.
    """
    self.__descaling_factor = float(self._get_parameter("descaling_factor"))

    self.__ft = self._get_parameter("forward_transformation")
    if self.__ft is None:
      self.__ft = lambda x: x

    self.__bt = self._get_parameter("backward_transformation")
    if self.__bt is None:
      self.__bt = lambda x: x

    self.__pixel_min = self._get_parameter("pixel_min")
    self.__pixel_max = self._get_parameter("pixel_max")
    self.__add_uniform = self._get_parameter("dediscretization") == "uniform"
    self.__base_data = self._get_parameter("base_data")

    self.__discretize = self._get_parameter("discretize")

    self.__denormalize = self._get_parameter("denormalize")
    # difference between 2 pixels in end result data
    # used in denormalization
    self.__pixel_diff = ( 1 / self.__descaling_factor
                          if not self.__denormalize
                          else 1.)

    self.__retain_shape = self._get_parameter("retain_shape")

  def __len__(self):
    self.ensure_available()
    return len(self.__base_data)

  def __getitem__(self, idx):
    self.ensure_available()

    with torch.no_grad():
      if self.__data is not None:
        _, label = self.__base_data[idx]

        val = self.__data[idx]

        if self.__mmap:
          val = torch.from_numpy(val.copy())
      else:
        val, label = self.__base_data[idx]

        val = from_batch(self.__static_transform(to_batch(val)))

      if self.__add_uniform:
        val += torch.empty_like(val).uniform_(0., self.__pixel_diff)

    return (val, label)

  @staticmethod
  def make_mmap(mode, path, meta_data):
    return np.memmap(
      BacktransformedData.__make_mmap_data_path(path),
      mode=mode,
      shape=meta_data["data_shape"],
      dtype=meta_data["data_dtype"]
    )

  def _load(self, path):
    self.__initialize()

    if is_static_set(self.__base_data):
      # the data will have been saved
      if self.__mmap and not os.path.exists(self.__make_meta_path(path)):
        self.__data = self.__make_transformed_data(path)
      elif (not self.__mmap
          and not os.path.exists(self.__make_data_path(path))):
        self.__data = self.__make_transformed_data(path)
        self.__save_data(path)
      else:
        if self.__mmap:
          with open(self.__make_meta_path(path), "rb") as file:
            meta_data = pickle.load(file)
          self.__data = self.make_mmap("r", path, meta_data)
        else:
          self.__data = torch.load(self.__make_data_path(path))
    else:
      self.__data = None

    return super()._load(path)

  def __static_transform(self, inputs):
    """
    Performs the static part of the dataset transformation
    """
    with torch.no_grad():
      outputs = self.__bt(self.__ft(inputs))

      outputs *= self.__descaling_factor

      if self.__discretize:
        outputs.round_()

      if self.__pixel_min is not None or self.__pixel_max is not None:
        outputs = torch.clamp(outputs, min=self.__pixel_min, max=self.__pixel_max)

      if not self.__denormalize:
        outputs /= self.__descaling_factor

      if self.__retain_shape:
        outputs = utils.upshape(outputs, inputs.shape[1:])

    return outputs

  def __make_transformed_data(self, path=None):
    """
    Returns the statically transformed data, not including random
    dediscretization.
    """
    data = None

    base_data = self.__base_data

    data_length = len(base_data)

    if len(base_data) > 0 and self.__batch_size > 1:
      input_shape = base_data[0][0].shape

    for start in range(0, data_length, self.__batch_size):
      end = min(start + self.__batch_size, data_length)

      if self.__batch_size == 1:
        inputs = to_batch(base_data[start][0])
      else:
        inputs = torch.empty((end - start,) + input_shape, dtype=base_data[start][0].dtype)
        for i in range(end - start):
          inputs[i], _ = base_data[start + i]

      outputs = self.__static_transform(inputs)

      if data is None:
        data_shape = (data_length,) + outputs.shape[1:]
        if self.__mmap:
          meta_data = {}
          meta_data["data_shape"] = data_shape
          meta_data["data_dtype"] = outputs.cpu().numpy().dtype

          data = self.make_mmap("w+", path, meta_data)
        else:
          data = torch.empty(data_shape, dtype=outputs.dtype)

      if self.__mmap:
        outputs = outputs.cpu().numpy()

      data[start:end] = outputs

      progress_bar(start // self.__batch_size + 1,
        batch_count(len(base_data), self.__batch_size),
        pre_text=" Computing Dataset ")

    if self.__mmap:
      with open(self.__make_meta_path(path), "wb") as file:
        pickle.dump(meta_data, file)

      data.flush()
      data = self.make_mmap("r", path, meta_data)

    return data

  def _unloadable(self):
    return True

  def unload(self):
    try:
      del self.__descaling_factor, self.__ft, self.__bt, \
        self.__pixel_min, self.__pixel_max, \
        self.__add_uniform, self.__base_data, \
        self.__discretize, self.__denormalize, \
        self.__pixel_diff, self.__retain_shape, \
        self.__data
    except AttributeError:
      pass
    
    super().unload()

  def _produce(self):
    self.__initialize()

class SampleInterleavedData(VirtualReproducible):
  """
  Data from a dataset that has been interleaved with randomly generated data
  from some distribution that implements a sample() method. The labels of the
  data will be 0 for data from the dataset and 1 for random data.
  """
  def __init__(self, parameters={},
    device=None,
    sample_on_device=False,
    **kwargs):
    """
    Args:
      device: Device that data should be created on or sent to before being
        emitted. Used to ensure that tensors from both sources are on the same
        device. E.g. torch.device("cuda") or torch.device("cpu").
      sample_on_device: If set, the device parameter will be passed to the
        sampling function.
      parameters: Dictionary of parameters to set in the reproducible.
        Parameters for SampleInterleavedData are:
        - dataset: Dataset which should be interleaved with random data.
          The dataset is expected to return a tuple (data, label).
        - sample_distribution: Distribution which to sample from. Must
          implement a sample() method that accepts batch_size as a positional
          parameter and doesn't require other parameters.
        - data_count: Number of items from the dataset that will be returned
          in a row before data is sampled from the distribution.
        - samples_count: Number of samples from the distribution that will be
          sampled in a row before data from the dataset is used again.
    """
    parameters = {**{
        "_SampleInterleavedData__version": "1.0.0",
        "dataset": None,
        "sample_distribution": None,
        "data_count": 1,
        "samples_count": 1
      },
      **parameters}

    self.__device = device
    self.__sample_on_device = sample_on_device

    self.__sample_arguments = {}
    if sample_on_device:
      self.__sample_arguments["device"] = device

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__data = self._get_parameter("dataset")
    self.__sampler = self._get_parameter("sample_distribution")
    self.__ndata = int(self._get_parameter("data_count"))
    self.__nrand = int(self._get_parameter("samples_count"))
    self.__len = (
      len(self.__data) // self.__ndata) * (self.__nrand + self.__ndata)

  def __data(self):
    return self._get_parameter("dataset")

  def __len__(self):
    self.ensure_available()
    return self.__len

  def __getitem__(self, idx):
    self.ensure_available()
    check_inrange(self, idx, "interleaved data")

    batch_size = self.__ndata + self.__nrand

    idx_local = idx % batch_size
    if idx_local >= self.__ndata:
      # random sample
      data = self.__sampler.sample(1, **self.__sample_arguments)[0]
      label = 1
    else:
      data, _ = self.__data[(idx // batch_size) * self.__ndata + idx_local]
      label = 0

    if self.__device is not None:
      data = data.to(self.__device)

    return (data, label)

class RandomSingleColorImages(VirtualReproducible):
  """
  Produce images with a random color. Color is uniformly distributed across the
  RGB bit color space. Images are in 
  """
  def __init__(self, parameters={}, **kwargs):
    """
    Args:
      parameters: Dictionary of parameters to set in the reproducible.
        Parameters for RandomSingleColorImages are:
        - epoch_size: Length of the data for training purposes.
        - width: Width of the returned images.
        - height: Height of the returned images.
        - "dediscretization": None or "uniform". Uniform dediscretization adds
          a uniform distribution on [0,1] per pixel.
    """
    parameters = {**{
        "_RandomSingleColorImages__version": "1.0.0",
        "epoch_size": 10000,
        "width": None,
        "height": None,
        "dediscretization": None,
      },
      **parameters}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__len = self._get_parameter("epoch_size")
    self.__width = self._get_parameter("width")
    self.__height = self._get_parameter("height")
    self.__dedisc = self._get_parameter("dediscretization")

  def __len__(self):
    self.ensure_available()
    return self.__len

  def __getitem__(self, idx):
    self.ensure_available()
    check_inrange(self, idx, "RandomSingleColorImages")

    data = torch.empty(3,self.__height,self.__width)
    data[0] = torch.randint(0,256,(1,))
    data[1] = torch.randint(0,256,(1,))
    data[2] = torch.randint(0,256,(1,))

    data = data.float()

    return (data,0)

class PreprocessedImages(VirtualReproducible):
  """
  Preprocess image data.
  """
  def __init__(self, inputs, parameters={}, **kwargs):
    """
    Args:
      inputs: Input data.
      parameters: Dictionary of parameters to set in the reproducible.
        Parameters for RandomSingleColorImages are:
        - "input": Input data, should be a dataset of images in CHW format versus
          labels.
        - "dediscretization": None or "uniform". Uniform dediscretization adds
          a uniform distribution on [0,1] per pixel.
        - "dediscretization_transformation": Matrix tensor that will be right
          multiplied with the discretization summand (interpreted as vector)
          before adding it to the data.
        - "shift": Additive componentwise shift to all pixel data.
        - "pre_factor": The factor the data would need to be multiplied
          with before other transformations like dediscretization.
        - "post_factor": The factor that data will be multiplied with after
          other transformations.
        - "output_start": Start outputting data at this original element.
        - "output_length": Cut or repeat data to get this total length. None
          for original length.
    """
    parameters = {**{
        "_PreprocessedImages__version": "1.0.4",
        "dediscretization": None,
        "pre_factor": None,
        "post_factor": None,
        "shift": None,
        "output_start": 0,
        "output_length": None,
        "dediscretization_transformation": None,
      },
      **parameters,
      **{
        "input": inputs,
    }}

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__inputs = self._get_parameter("input")
    self.__dedisc = self._get_parameter("dediscretization")
    
    if self.__dedisc is not None:
      self.__dedisc_transform = self._get_parameter("dediscretization_transformation")
      self.__data_dimensions = np.prod(self.__inputs[0][0].shape)

    self.__pre_factor = self._get_parameter("pre_factor")
    self.__post_factor = self._get_parameter("post_factor")
    self.__shift = self._get_parameter("shift")
    self.__start = self._get_parameter("output_start")
    self.__len = self._get_parameter("output_length")
    if self.__len is None:
      self.__len = len(self.__inputs)

  def __len__(self):
    self.ensure_available()
    return self.__len

  def __getitem__(self, idx):
    self.ensure_available()
    check_inrange(self, idx, "PreprocessedImages")

    idx = (self.__start + idx) % len(self.__inputs)

    data, label = self.__inputs[idx]
    if data.dtype != torch.float and data.dtype != torch.double:
      data = data.type(torch.float)

    if self.__pre_factor is not None:
      data *= self.__pre_factor

    if self.__dedisc == "uniform":
      if self.__dedisc_transform is None:
        with torch.no_grad():
          data += torch.empty_like(data).uniform_(0., 1.)
      else:
        with torch.no_grad():
          dedisc_summand = torch.empty((self.__data_dimensions,)).uniform_(0., 1.)
          dedisc_summand = torch.matmul(self.__dedisc_transform, dedisc_summand)
          data += dedisc_summand.view(data.shape)

    if self.__shift is not None:
      data += self.__shift

    if self.__post_factor is not None:
      data *= self.__post_factor

    return (data,label)

  def is_static(self):
    """
    Returns whether this dataset is static, i.e. doesn't change.
    """
    return is_static_set(self._get_parameter("input")) and \
      self._get_parameter("dediscretization") is None

class SceneDataset(VirtualReproducible):
  def __init__(self, scene_description_path, root="scene", parameters={}, **kwargs):
    self.__path = Path(scene_description_path).expanduser()

    with self.__path.open() as f:
      input_dict = yaml.safe_load(f)

    parameters = {**{
        "_SceneDataset__version": "1.0.2",
        "scene_description": input_dict,
        "root_object": root,
        "epoch_size": 10000,
        "normalized": False
      },
      **parameters
    }

    super().__init__(parameters=parameters, **kwargs)

  def _produce(self):
    self.__scene = scene_generator.read_object_dict(
      self._get_parameter("scene_description"), self.__path.parent
    )[self._get_parameter("root_object")]

    self.__len = self._get_parameter("epoch_size")
    self.__to_torch = tv.transforms.ToTensor()
    self.__factor = 255 if not self._get_parameter("normalized") else 1.

  def __len__(self):
    self.ensure_available()
    return self.__len

  def __getitem__(self, idx):
    self.ensure_available()
    if idx >= self.__len:
      raise IndexError("Index out of range.")

    # Annotations about included base images could be used as label if desired
    return self.__to_torch(self.__scene.get().convert("RGB")) * self.__factor, 0

  def max_entropy(self):
    self.ensure_available()
    return self.__scene.max_entropy()

  def fully_distinguishable(self):
    self.ensure_available()
    return self.__scene.is_distinguishable(np.zeros((0,0), dtype=np.bool))
