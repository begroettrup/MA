import pickle
import hashlib
import os.path
from os import makedirs
import shutil
from inspect import getmodule

import yaml

from weakref import WeakSet

from .parametrizable import Parametrizable

def _typeName(obj):
  """
  Returns a qualified name for the given type, class or function.
  """
  try:
    qualname = obj.__qualname__
  except AttributeError:
    qualname = obj.__name__
  if qualname == "<lambda>":
    return None
  if obj.__module__ != "__main__":
    name = obj.__module__ + '.'
  else:
    try:
      name = os.path.splitext(os.path.basename(getmodule(obj).__file__))[0] + '.'
    except AttributeError:
      name = ""
  return name + qualname

def maybe_reproduction(obj):
  """
  Calls reproduction() of the given object and passes through the return value
  if possible. If the object has no method reproduction() the object itself is
  returned.
  """
  try:
    r = obj.reproduction
  except AttributeError:
    return obj

  return r()


def _load_reproducible_options():
  base_options = {
    'base_dir': "~/.reproducibles"
  }

  try:
    with open(os.path.expanduser("~/.reproducibles.yml"), 'r') as f:
      user_options = yaml.safe_load(f)
  except FileNotFoundError:
    user_options = {}

  for option_name in user_options.keys():
    if option_name not in base_options.keys():
      raise ValueError("Unknown option " + option_name +
        " in ~/.reproducibles.yml")

  return {**base_options, **user_options}

class Reproducible(Parametrizable):
  __options = _load_reproducible_options()
  __base_directory = __options['base_dir']
  __references = WeakSet()

  """
  A reproducible is an object which:
   - Has a number of parameters.
   - Is produced via a (semi-)reproducible method, depending on its parameters.
     Two Reproducibles with the same parameters should be equivalent.
   - Can be saved and loaded.

  A subclass must implement _produce() and call super().__init__() accordingly.
  It can access its parameters through _parameters.
  _save() and _load() per default will use pickle to persist the value returned
  by reproduction(). Specializations may be implemented to improve performance
  or save additional information about the Reproducible.
  """
  def __init__(self, parameters={}, **kwargs):
    """
    Create a new object with the given parameters.

    Args:
      base_directory: Base directory under which this reproducible will be saved.
    """
    parameters = {**{
      "_Reproducible_version": "3.0.0"
    }, **parameters}

    self.__initialized = False
    self.__production_counter = 0

    self.__references.add(self)
    self.__on_available = []

    super().__init__(parameters=parameters, **kwargs)

  @staticmethod
  def set_base_directory(base_directory):
    """
    Sets the global base directory for Reproducibles. Must be called before any
    Reproducible objects are initialized.
    """
    Reproducible.__base_directory = base_directory

    for r in Reproducible.__references:
      r._delete_cache()

  def _delete_cache(self):
    """
    Delete all cached values that depend on the parameters of the object.
    """
    self.__initialized = False
    try:
      del self.__val
    except AttributeError:
      pass
    try:
      del self.__save_directory
    except AttributeError:
      pass
    try:
      del self.__pparams
    except AttributeError:
      pass
    super()._delete_cache()

  def _get_parameter(self, name):
    """
    Internal parameter access.
    """
    return maybe_reproduction(super()._get_parameter(name))

  def _parameter_names(self):
    """
    Internal routine to get parameter names.
    """
    return super().parameter_names()

  @staticmethod
  def __make_parameters_file_path(base_path):
    """
    Given a path to the directory for a specific reproducible object instance,
    return the path to the parameters file within it.
    """
    return os.path.join(base_path, "parameters.pk")

  def _unloadable(self):
    return False

  def _unload_parameters(self):
    for p in self.get_parameters():
      try:
        p.unload()
      except AttributeError:
        pass

  def unload(self):
    if self._unloadable:
      self._unload_parameters()
      try:
        del self.__val
      except AttributeError:
        pass

  def __get_parameters_file_path(self):
    """
    Return the path to the file in which the parameters of this Reproducible are
    be stored.
    """
    return Reproducible.__make_parameters_file_path(self.__get_save_dir())

  @staticmethod
  def __make_artifact_path(base_path):
    """
    Given a path to the directory for a specific reproducible object instance,
    return the path to the artifacts folder within it.
    """
    return os.path.join(base_path, "artifacts")

  def __get_artifact_path(self):
    """
    Return the path to the directory containing the persistent representation
    of the Reproducible.
    """
    return Reproducible.__make_artifact_path(self.__get_save_dir())

  def __get_save_dir(self):
    """
    Returns the name of the directory that will be used for saving this
    Reproducible with the current parameters.
    """
    try:
      return self.__save_directory
    except AttributeError:
      # hash the current parameters to find corresponding directory
      parameters_hash = self.parameters_hash()
      base_path = os.path.expanduser(self.__base_directory)
      # This will return the name of whatever subtype the current object has
      obj_name = _typeName(type(self))
      i = 0
      while True:
        path = os.path.join(base_path, obj_name, parameters_hash + "-" + str(i))
        parameters_path = Reproducible.__make_parameters_file_path(path)
        if os.path.exists(parameters_path):
          # somebody created this object with the same version hash before
          # this almost surely means this path is correct, but we make sure
          with open(parameters_path, "rb") as parameters_file:
            if parameters_file.read() == self.__parameters_dump():
              break
        else:
          # delete the directory in case it exists but was corrupted
          if os.path.exists(path):
            shutil.rmtree(path)
          makedirs(Reproducible.__make_artifact_path(path))
          # there was no existing directory for this parameters combination
          break
        i += 1

      self.__save_directory = path

    return self.__save_directory

  def __persistable_parameters(self):
    """
    Returns a version of the parameters that can be persisted to disk.
    """
    try:
      return self.__pparams
    except AttributeError:
      self.__pparams = {}
      for key in self._parameter_names():
        value = super().get_parameter(key)
        try:
          value = value.parameters_hash()
        except AttributeError:
          pass
        self.__pparams[key] = value
      return self.__pparams

  def __parameters_dump(self):
    """
    Produces a pickle dump of the current parameters.
    """
    return pickle.dumps(self.__persistable_parameters())

  def parameters_hash(self):
    """
    Produces a hash string that encodes the options of this object.
    """
    return hashlib.sha256(self.__parameters_dump()).hexdigest()

  def _is_reproducing(self):
    """
    Checks whether a reproduction process is currently under way.
    """
    return self.__production_counter > 0

  def _reproduction_depth(self):
    """
    Returns the recursion depth of the current reproduction process. This is 0
    if no reproduction is active and 1 if there is no reproduction recursion
    but an ongoing reproduction. Otherwise the value is equal to the number of
    recursive reproduction calls including the initial one.

    A reproduction process is any call to reproduction() or reproduce_value().
    That call may end in either a _load() or a _produce().
    """
    return self.__production_counter

  def ensure_available(self):
    """
    Triggers reproduction if it is not currently underway.
    """
    if not self._is_reproducing():
      self.reproduction()

  def reproduction(self):
    """
    Produces the object and buffers the result. If the object represents a
    computation, the result of that computation will be returned, otherwise the
    whole object will be.

    Note that object parameters need to be set again if they change to trigger
    a new computation.
    """
    ret = self.reproduce_value()
    if ret is None:
      return self
    else:
      return ret

  def _set_dependent_parameters(self):
    """
    This is called whenever a reproduction is requested. If the values of some
    parameters depend on others, set them here.
    """
    pass

  def __is_available(self):
    try:
      v = self.__val
      return True
    except AttributeError:
      return not self._is_reproducing()

  def __run_hooks(self):
    for f in self.__on_available:
      f(self)

  def reproduce_value(self):
    """
    Produces the object and buffers and returns the result.
    """
    try:
      return self.__val
    except AttributeError:
      self._set_dependent_parameters()
      self.__production_counter += 1
      # load the object if possible
      artifact_path = self.__get_artifact_path()
      parameters_path = self.__get_parameters_file_path()
      if os.path.exists(parameters_path):
        self.__val = self._load(artifact_path)
      else:
        self.__val = self._produce()
        # save the object
        self._save(self.__val, artifact_path)
        # save current parameters into pickle
        with open(parameters_path, "wb") as parameters_file:
          pickle.dump(self.__persistable_parameters(), parameters_file)
      self.__production_counter -= 1
      if not self._is_reproducing():
        self.__run_hooks()

    return self.__val

  # the following three methods should be implemented by the subclass
  def _produce(self):
    """
    Recreate this object from scratch, using the parameters set in _parameters.

    The return value of this method will be buffered and returned in
    reproduction().
    """
    raise NotImplementedError()

  @staticmethod
  def __make_value_path(path):
    """
    Returns the subpath of path to save the pickle representation of the object
    value under.
    """
    return os.path.join(path, "_Reproducible__value.pk")

  def _save(self, value, path):
    """
    Produce a persistent representation of the current state under the given
    path.

    By default uses pickle to save and restore the value of the .

    Args:
      value: The buffered value as returned by reproduction().
      path: Path to an existing folder in which the object representation
        should be saved. To enable cooperative calls, the representation should
        be saved in a single file or subfolder of that path with a subclass
        specific name.
        The same path will be passed to _load() to restore an old state.
      base_directory: Base directory under which this reproducible will be
        saved.
    """
    with open(Reproducible.__make_value_path(path), "wb") as file:
      pickle.dump(value, file)

  def _load(self, path):
    """
    Restore the object state from a persistent file which was created by a call
    to _save. The return value should be the same as that of the call to
    _produce which created the saved object.

    Args:
      path: Path to the folder from which to load the value representation.
    """
    with open(Reproducible.__make_value_path(path), "rb") as file:
      return pickle.load(file)

  def once_available(self, f):
    """
    Sets a function that is called whenever the object becomes available or
    immediately if it already is. This method is preferred over modifying the
    reproducible object since it makes sure the state is reinstated after
    unloading.
    """
    self.__on_available.append(f)
    if self.__is_available():
      f(self)

class ReproducibleFunction(Reproducible):
  def __init__(self, function, args=[], kwargs={}, name=None,
    parameters={}, **other):
    """
    Args:
      function: Function to execute when producing the value.
      name: A unique name to identify this function. If None will be
        set to the qualified name of this 
      args: List of default values for positional arguments to the
        function. These will be added as indexed parameters to the Reproducible
        according to their position.
      kwargs: Dictionary of keyword arguments to the function with default
        values.
      parameters: Additional parameters to add to the function reproducible.
        These will be merged with the parameters from args and kwargs.
    """
    parameters = parameters.copy()

    if name is None:
      name = _typeName(function)
      if name is None:
        raise ValueError("Reproducible lambda functions must be named.")

    # Identifier for this function -- Reproducibles of the same class are only
    # differentiated by their parameters
    parameters["_ReproducibleFunction_name"] = name

    for key, value in kwargs.items():
      if type(key) != str:
        raise ValueError("Keyword argument names must be strings.")
      parameters[key] = value

    for key, value in enumerate(args):
      parameters[key] = value

    super().__init__(
      parameters=parameters,
      **other)

    self.__function = function
    self.__nargs = len(args)
    self.__argument_keywords = kwargs.keys()

  def _produce(self):
    args = []
    for i in range(self.__nargs):
      args.append(self._get_parameter(i))

    kwargs = {}
    for key in self.__argument_keywords:
      kwargs[key] = self._get_parameter(key)

    return self.__function(*args, **kwargs)

class CallableReproducible(Reproducible):
  """
  This class can be used instead of Reproducible when a subclass which is
  later in the serialization of the inheritance order is callable.

  It ensures that calls to the given object always happen on the reproduced
  variant.
  """
  def __call__(self, *args, **kwargs):
    self.ensure_available()
    return super().__call__(*args, **kwargs)

class VirtualReproducible(Reproducible):
  """
  A reproducible that doesn't safe data but just acts as an interface for
  parameters.
  """
  def _load(self, path):
    return self._produce()

  def _save(self, value, path):
    pass

  def _produce(self):
    return None
