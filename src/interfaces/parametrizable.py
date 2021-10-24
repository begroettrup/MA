class Parametrizable:
  """
  An object that has a set of parameters that can be changed and accessed. The
  set of parameters is fixed, only their value changes.

  Parameters can be accessed via get_parameter() and set_parameter().

  Specializations should override _delete_cache() accordingly such that any
  parameter dependent cached values are deleted through that method.
  """
  def __init__(self, parameters={}, **kwargs):
    """
    Args:
      parameters: Parameters for this object. Should be a dictionary of parameter
        name to default value.
    """
    self.__parameters = parameters.copy()
    super().__init__(**kwargs)

  def __get_parameter(self, name):
    return self.__parameters[name]

  def get_parameter(self, name):
    """
    Public interface to get the value of a parameter.
    """
    return self.__get_parameter(name)

  def __delete_cache(self):
    """
    Resets the current object state by calling _delete_cache() and checks that
    the _delete_cache() call was propagated correctly.
    """
    self.__invalidated = True
    self._delete_cache()
    if self.__invalidated:
      raise RuntimeError("_delete_cache call was not propagated correctly. "
        "All classes implementing Parametrizable should use super() to "
        "propagate calls to _delete_cache().")

  def __set_parameter(self, name, value):
    # invalidate existing result if parameters change
    if self.__parameters[name] != value:
      self.__delete_cache()
      self.__parameters[name] = value

  def _add_parameter(self, name, value):
    """
    Adds a new parameter to the Parametrizable if the parameter wasn't present
    before.
    """
    if name in self.__parameters:
      raise ValueError("Can't add parameter " + name +
                       ": Parameter already exists.")
    self.__delete_cache()

  def _delete_parameter(self, name):
    """
    Removes a parameter from the set of parameters if it was part of it.
    """
    if name in self.__parameters:
      del self.__parameters[name]
      self.__delete_cache()

  def set_parameter(self, name, value):
    """
    Public interface to set a parameter.
    """
    return self.__set_parameter(name, value)

  def get_parameters(self):
    """
    Returns a dictionary containing all (public) parameters that this
    Parametrizable has.
    """
    return dict([(param, self.get_parameter(param))
      for param in self.parameter_names()])

  def _get_parameter(self, name):
    """
    Internal interface for getting parameters through subclasses.
    """
    return self.__get_parameter(name)

  def _set_parameter(self, name, value):
    """
    Internal interface for setting parameters.
    """
    return self.__set_parameter(name, value)

  def set_parameters(self, param_dict):
    """
    Sets all parameters with key in param_dict to their corresponding value.
    """
    for key, val in param_dict.items():
      self.set_parameters(key, val)

  def parameter_names(self):
    """
    Returns the names of all parameters this Parametrizable has.
    """
    return self.__parameters.keys()

  def _delete_cache(self):
    """
    Delete any cached values that depend on the parameters of the object.
    """
    self.__invalidated = False
    pass

class ParameterDescription:
  """
  A class describing a parameter a reproducible exposes to the outside.

  Args:
    description: Text description of the parameter. Will be saved in
      self.description.
    type_: Type of the parameter that should be used when parsing strings, for
      example command line arguments, into it.
    optional: Whether the parameter may be None.
    options: List of all possible options for this parameter. None means no
      restriction.
  """
  def __init__(self, description="", type_=None, optional=False, options=None):
    self.description = description
    self.type = type_
    self.optional = optional
    self.options = options

class ParameterInterface(Parametrizable):
  """
  A Parametrizable with an interface which hides private parameters and
  describes public ones.
  """
  def __init__(self, parameters={}, parameter_infos={}, **kwargs):
    """
    Args:
      parameter_infos: ParameterDescriptions for any parameters that should be exposed.
        These infos may be empty, if the desire is just to expose a parameter.
    """
    for key, value in parameter_infos.items():
      if key not in parameters:
        raise ValueError(key + " is not a parameter of the Parametrizable. "
          "Argument infos must correspond to existing parameters.")

    self.__parameter_infos = parameter_infos.copy()
    super().__init__(parameters=parameters, **kwargs)

  def parameter_names(self):
    """
    Returns the names of all public parameters this Parametrizable has.
    """
    return self.__parameter_infos.keys()

  def parameter_info(self, name):
    """
    Returns meta information about a public parameter of the Parametrizable.
    """
    return self.__parameter_infos[name]

  def get_parameter(self, name):
    if name in self.__parameter_infos:
      return super().get_parameter(name)
    else:
      raise AttributeError('Object does not have a public parameter "'
        + name + '"')

  def set_parameter(self, name, value):
    if name in self.__parameter_infos:
      # invalidate existing result if parameters change
      super().set_parameter(name, value)
    else:
      raise AttributeError('Object does not have a public parameter "'
        + name + '"')

def make_empty_description(value):
  """
  Creates an empty description for an argument with the given default value.
  """
  if value is not None:
    return ParameterDescription(type_=type(value))
  return ParameterDescription()
