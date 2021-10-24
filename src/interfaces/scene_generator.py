import yaml

from PIL import Image

from pathlib import Path

import random
import math

import itertools

import numpy as np

def assert_is_type(element, type_, location):
  if element.type() != type_:
    raise TypeError("Type '" + element.type() + "' of element '" + element.name + "' "
      "does not match previously inferred type '" + type_ + "' within " + location +
      "!")

def area_union(areas):
  """
  Given an iterable of area masks returns the union of them, that is an area
  mask which has every pixel set which is set in any of the masks.
  """
  areas = list(areas)

  # hwc
  height = max(map(lambda x: x.shape[0], areas))
  width = max(map(lambda x: x.shape[1], areas))

  # initialize empty
  union = np.zeros((height, width), dtype=np.bool)

  for a in areas:
    union[:a.shape[0],:a.shape[1]] |= a[:,:]

  return union

def area_intersection(areas):
  """
  Returns the intersection of a number of area masks. That is an area mask that
  is true where all of the areas are true.
  """
  areas = list(areas)

  # hwc
  height = min(map(lambda x: x.shape[0], areas))
  width = min(map(lambda x: x.shape[1], areas))

  # initialize full
  intersection = np.ones((height, width), dtype=np.bool)

  for a in areas:
    intersection[:,:] &= a[:height,:width]
  
  return intersection

def enlarge_mask(target_shape, mask, fill):
  h = min(target_shape[0], mask.shape[0])
  w = min(target_shape[1], mask.shape[1])

  new_mask = np.full(target_shape[:2], fill)

  new_mask[:h,:w] = mask[:h,:w]

  return new_mask

def masked_area(area, mask, fill=False):
  return area[enlarge_mask(area.shape, mask, fill)]

def trace_cause_on_fail(condition, failure_msg):
  if SceneGenerator.distinction_failure_message is None and not condition:
    SceneGenerator.distinction_failure_message = failure_msg

  return condition

def trace_distinction_failure_condition(condition, msg):
  if SceneGenerator.distinction_failure_message is None and condition:
    SceneGenerator.distinction_failure_message = msg
    return True
  return False

def trace_first_empty_identifiable_area(id_area, msg):
  trace_distinction_failure_condition((~id_area).all(), msg)

  return id_area

class BaseImage:
  def __init__(self, name, path_str, base_path):
    self.name = name

    path = Path(path_str).expanduser()

    if not path.is_absolute():
      path = base_path / path

    self.__path = path

    # test whether image is valid by opening it and closing it
    if not path.exists() or not path.is_file():
      raise ValueError(str(path) + " is not a valid file!")

  def get(self):
    return Image.open(self.__path)

  def choices(self):
    return [self.get()]

  def __asarray(self):
    return np.asarray(self.get().convert("RGBA"))

  def type(self):
    return "Image"

  def n_possibilities(self):
    return 1

  def max_entropy(self):
    return 0

  def inner_size(self):
    return self.__asarray().shape[:2]

  def bound_size(self):
    return self.inner_size()

  def covered_area(self):
    # anything where this has non 0 alpha value is considered covered
    # since this image might impact the final value of that position
    return self.__asarray()[:,:,3] != 0

  def identifiable_area(self, covered_area):
    # only fully opaque spots are considered as identifiable
    # since otherwise colors might be changed by the background image
    # covered spots are also not identifiable
    arr = self.__asarray()

    not_covered = enlarge_mask(arr.shape, ~covered_area, True)

    return trace_first_empty_identifiable_area(
      (arr[:,:,3] == 255) & not_covered,
      "Identifiable pixels of BaseImage " + self.name + " were fully covered."
    )

  def identifying_values(self, identifying_area):
    return [ masked_area(self.__asarray()[:,:,:3], identifying_area) ]

  def is_distinguishable(self, covered_area):
    # always distinguishable: there is only a single value
    return True

class Left:
  def __init__(self, name, val):
    self.name = name

    if type(val) != int:
      raise ValueError("Left value " + name + " must be an int and wasn't!")

    self.__val = val

  def get(self):
    return self.__val

  def type(self):
    return "Left"

  def choices(self):
    return [self.__val]

  def n_possibilities(self):
    return 1

  def max_entropy(self):
    return 0

class Top:
  def __init__(self, name, val):
    self.name = name

    if type(val) != int:
      raise ValueError("Top value " + name + " must be an int and wasn't!")

    self.__val = val

  def get(self):
    return self.__val

  def type(self):
    return "Top"

  def choices(self):
    return [self.__val]

  def n_possibilities(self):
    return 1

  def max_entropy(self):
    return 0

class Choice:
  def __init__(self, name, choices: list) -> None:
    self.name = name

    if len(choices) == 0:
      raise ValueError("Choice " + self.name + " must include at least one element!")
    
    self.__type = choices[0].type()

    for c in choices:
      if self.__type is None:
        self.__type = c.type()
      else:
        assert_is_type(c, self.__type, self.name)

    self._choices = choices

  def inner_size(self):
    h, w = self._choices[0].inner_size()

    for c in self._choices[1:]:
      h_other, w_other = c.inner_size()
      h = min(h, h_other)
      w = min(w, w_other)

    return h, w

  def bound_size(self):
    h, w = self._choices[0].bound_size()

    for c in self._choices[1:]:
      h_other, w_other = c.bound_size()
      h = max(h, h_other)
      w = max(w, w_other)

    return h, w

  def get(self):
    return random.choice(self._choices).get()

  def type(self):
    return self.__type

  def n_possibilities(self):
    # the number of possibilities does not take into account whether all choices lead to
    # different result
    return sum(map(lambda x: x.n_possibilities(), self._choices))

  def max_entropy(self):
    """
    Maximum entropy assuming that our choices all matter.
    """
    # uses the formula for subchoices: entropy increases by entropy of the additional choice
    # times its probability
    return math.log(len(self._choices)) \
      + sum(map(lambda x: x.max_entropy(), self._choices)) / len(self._choices)

  def covered_area(self):
    # covered area is the union of all covered areas
    return area_union(map(lambda x: x.covered_area(), self._choices))

  def identifiable_area(self, covered_area):
    # identifiable area is the intersection of all identifiable areas

    return trace_first_empty_identifiable_area(
      area_intersection(map(lambda x: x.identifiable_area(covered_area),
                                 self._choices)),
      "Identifiable pixels of elements in Choice " + self.name + " had empty intersection."
    )

  def identifying_values(self, identifying_area):
    # identifying values are all areas that identify any of the elements in the choice
    all_idvs = []

    for id_values in map(lambda x: x.identifying_values(identifying_area), self._choices):
      all_idvs += list(id_values)

    return np.unique(np.stack(all_idvs), axis=0)

  def is_distinguishable(self, covered_area):
    # the choice is distinguishable if each of the possibilities is and they can also be
    # differentiated among each other by having different identifying values in the
    # identifying area of the choice
    id_area = self.identifiable_area(covered_area)

    if not id_area.any():
      # empty identifying area -> can't distinguish anything
      return False

    return trace_cause_on_fail(
      all(map(lambda x: x.is_distinguishable(covered_area), self._choices)) and (
      np.unique(
        self.identifying_values(id_area), axis=0, return_counts=True)[1] == 1
      ).all(),
      "Not all subobjects of Choice " + self.name + " could be distinguished."
    )

  def choices(self):
    return [ c for o in self._choices for c in o.choices() ]

  def choices_by_type(self, type_):
    return [ c for o in self._choices for c in o.choices_by_type(type_) ]

class ProportionalChoice (Choice):
  """
  A choice which selects elements proportional to exponential of their entropy such that
  the resulting entropy is maximized.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__proportions = list(map(lambda x: math.exp(x.max_entropy()), self._choices))

  def get(self):
    return random.choices(self._choices, weights=self.__proportions)[0].get()

  def max_entropy(self):
    entropy = 0
    total_weight = sum(self.__proportions)
    for choice, proportion in zip(self._choices, self.__proportions):
      p = proportion / total_weight
      entropy += -p*math.log(p) + p*choice.max_entropy()

    return entropy

class CombinedShift:
  def __init__(self, name, shifts):
    self.name = name
    if len(shifts) == 0:
      raise ValueError("CombinedShift must contain at least 1 shift!")

    for s in shifts:
      assert_is_type(s, "Shift", self.name)

    self.__shifts = shifts

  def get(self):
    left, top = self.__shifts[0].get()

    for s in self.__shifts[1:]:
      dleft, dtop = s.get()
      left += dleft
      top += dtop

    return left, top

  def type(self):
    return "Shift"

  def n_possibilities(self):
    # assuming no shifts add up to the same combination
    return sum(map(lambda x: x.n_possibilities(), self.__shifts))

  def max_entropy(self):
    return math.log(self.n_possibilities())

  def choices(self):
    """
    Returns all shifts this could take. Includes duplicates if multiple combinations
    add to the same total shift.
    """
    all_shifts = np.array(list(self.__shifts[0].choices()))

    for s in self.__shifts[1:]:
      new_shifts = []
      for pshift in s.choices():
        new_shifts.append(all_shifts + pshift)
      all_shifts = np.concatenate(new_shifts)

    return list(all_shifts)

def translate_mask(mask, shift, fill=False):
  """
  Pixel (0,0) of returned mask will correspond to pixel shift of old one.
  """
  start_h_old = max(0, shift[0])
  start_w_old = max(0, shift[1])
  start_h_new = max(0, -shift[0])
  start_w_new = max(0, -shift[1])

  dh = mask.shape[0] - start_h_old
  dw = mask.shape[1] - start_w_old

  target_shape = (
    start_h_new+dh,
    start_w_new+dw
  )

  new_mask = np.full(target_shape, fill)

  new_mask[start_h_new:,start_w_new:] = \
      mask[start_h_old:start_h_old+dh,start_w_old:start_w_old+dw]

  return new_mask

class Union:
  def __init__(self, name, elems):
    self.name = name

    self.__elems = [elems[i] for x, i in sorted(zip(map(lambda x: x.type(), elems),
                                                    range(len(elems)))) ]

    types = self.__get_elem_types()
    for i in range(1,len(types)):
      if types[i-1] == types[i]:
        raise ValueError("All elements in the union " + self.name +
          " must have different types.")

  def __get_elem_types(self):
    return sorted(map(lambda x: x.type(), self.__elems))

  def get(self):
    return [ v for k, v in zip(self.__get_elem_types(),
                               map(lambda x: x.get(), self.__elems))]

  def __by_type(self, type_):
    for x in self.__elems:
      if x.type() == type_:
        return x
    raise ValueError("Union " + name + " does not contain an element of "
      "type " + type_ + ".")

  def get_by_type(self, type_):
    return self.__by_type(type_).get()

  def includes_type(self, type_):
    return type_ in self.__get_elem_types()

  def type(self):
    return "(" + ",".join(self.__get_elem_types()) + ")"

  def n_possibilities(self):
    return math.prod(map(lambda x: x.n_possibilities(), self.__elems))

  def max_entropy(self):
    return sum(map(lambda x: x.max_entropy(), self.__elems))

  def choices(self):
    def subchoices(es):
      if len(es) == 0:
        yield ()
      else:
        e = es[0]
        es = es[1:]
        for cs in subchoices(es):
          for c in e.choices():
            yield (c,) + cs

    return list(subchoices(self.__elems))

  def choices_by_type(self, type_):
    return self.__by_type(type_).choices()

  def covered_area(self):
    cov_area = self.__by_type("Image").covered_area()

    if self.includes_type("Shift"):
      covered_areas = []

      for s in self.__by_type("Shift").choices():
        covered_areas.append(translate_mask(cov_area, (-s[0],-s[1])))

      cov_area = area_union(covered_areas)
    
    return cov_area

  def __limit_size(self, size_getter, change_initiator):
    if self.includes_type("Shift"):
      im_size = size_getter(self.__by_type("Image"))

      limit_size = None

      for s in self.__by_type("Shift").choices():
        shifted_im_size = [max(0, im_size[0] + s[0]), max(im_size[1] + s[1], 0)]

        if limit_size is None:
          limit_size = shifted_im_size
        else:
          for i in [0,1]:
            if change_initiator(limit_size[i], shifted_im_size[i]):
              limit_size[i] = shifted_im_size[i]

      return tuple(limit_size)
    else:
      return size_getter(self.__by_type("Image"))

  def inner_size(self):
    return self.__limit_size(lambda x: x.inner_size(), lambda lim, new: lim > new)

  def bound_size(self):
    return self.__limit_size(lambda x: x.bound_size(), lambda lim, new: lim < new)

  def identifiable_area(self, covered_area):
    if self.includes_type("Shift"):
      id_areas = []
      
      for s in self.__by_type("Shift").choices():
        id_areas.append(translate_mask(
          self.__by_type("Image").identifiable_area(
            translate_mask(covered_area, s)),
          (-s[0],-s[1]), False))

      return area_intersection(id_areas)
    else:
      return self.__by_type("Image").identifiable_area(covered_area)

  def identifying_values(self, identifying_area):
    if self.includes_type("Shift"):
      # identifying values are all areas that identify any of the elements
      all_idvs = []

      im = self.__by_type("Image")

      for s in self.__by_type("Shift").choices():
        all_idvs += list(im.identifying_values(
          translate_mask(identifying_area, s)))

      return np.unique(all_idvs, axis=0)
    else:
      return self.__by_type("Image").identifying_values(identifying_area)

  def is_distinguishable(self, covered_area):
    if self.includes_type("Shift"):
      im = self.__by_type("Image")

      for s in self.__by_type("Shift").choices():
        if not im.is_distinguishable(translate_mask(covered_area, s)):
          return False

      return True
    else:
      return self.__by_type("Image").is_distinguishable(covered_area)

def enlarge_translate(mask, enlarge_shape, enlarge_fill,
    shift, translate_shape, translate_fill):
  return translate_mask_to_shape(enlarge_mask(enlarge_shape, mask, enlarge_fill),
                        shift, translate_shape, translate_fill)

def translate_mask_to_shape(mask, shift, target_shape, fill):
  """
  Returns the translated mask which fits the target shape and is shifted by the given
  amount, i.e. pixel (0,0) of the new mask corresponds to pixel (shift) of the old mask.
  """
  new_mask = np.full(target_shape, fill)

  start_h_old = max(0, shift[0])
  start_w_old = max(0, shift[1])
  start_h_new = max(0, -shift[0])
  start_w_new = max(0, -shift[1])

  dh = min(target_shape[0] - start_h_new, mask.shape[0] - start_h_old)
  dw = min(target_shape[1] - start_w_new, mask.shape[1] - start_w_old)

  new_mask[start_h_new:start_h_new+dh,start_w_new:start_w_new+dw] = \
      mask[start_h_old:start_h_old+dh,start_w_old:start_w_old+dw]

  return new_mask

class Overlay:
  def __init__(self, name, objects_with_shifts):
    self.name = name
    if len(objects_with_shifts) == 0:
      raise ValueError("Overlay must consist of at least one object!")

    assert_is_type(objects_with_shifts[0], "Image", self.name)
    for o in objects_with_shifts[1:]:
      assert_is_type(o, "(Image,Shift)", self.name)

    self.__objects_with_shifts = objects_with_shifts

  def inner_size(self):
    return self.__objects_with_shifts[0].inner_size()

  def bound_size(self):
    return self.__objects_with_shifts[0].bound_size()

  def get(self):
    img = None

    for o in self.__objects_with_shifts:
      if img is None:
        img = o.get().convert("RGBA")
      else:
        dimg, (left, top) = o.get()

        # crop into correct shape
        crop = (
          # left
          max(0, -left),
          # upper
          max(0, -top),
          # right
          min(dimg.width, img.width - left),
          # lower
          min(dimg.height, img.height - top)
        )
        if crop != (0,0,dimg.width,dimg.height):
          dimg = dimg.crop(crop)
        shift = left + crop[0], top + crop[1]
        img.alpha_composite(dimg.convert("RGBA"), shift)

    return img

  def type(self):
    return "Image"

  def choices(self):
    # would need to be all possible elements this could produce
    raise NotImplemented

  def n_possibilities(self):
    return math.prod(map(lambda x: x.n_possibilities(), self.__objects_with_shifts))

  def max_entropy(self):
    return sum(map(lambda x: x.max_entropy(), self.__objects_with_shifts))

  def covered_area(self):
    # union of all covered areas
    return area_union(list(map(lambda x: x.covered_area(), self.__objects_with_shifts)))

  def __max_id_area(self, covered_area):
    """
    Internal version that also returns the respective index.
    """
    # maximum identifiable area
    # start with none
    max_id_area = np.zeros((0,0), dtype=np.bool)
    n_identifiable = 0
    i_max = -1

    inner_size = self.inner_size()

    for i, ows in reversed(list(enumerate(self.__objects_with_shifts))):
      # start at top of step since that will be covering up former layers

      # reduce identifiable area to the min size of this picture
      ida = ows.identifiable_area(covered_area)[:inner_size[0]][:inner_size[1]]

      if ida.sum() > n_identifiable:
        max_id_area = ida
        i_max = i

      covered_area = area_union([
        covered_area,
        ows.convered_area()
      ])

    return i_max, max_id_area

  def identifiable_area(self, covered_area):
    return self.__max_id_area(covered_area)[1]

  def identifying_values(self, identifying_area):
    cover_mask = enlarge_mask(self.max_shape(), ~identifying_area, True)

    elem_id = self.__max_id_area(cover_mask)[0]

    if elem_id == -1:
      # no element identifies here...
      return [np.empty((0,3), dtype=np.uint8)]

    return self.__objects_with_shifts[elem_id].identifying_values(identifying_area)

  def is_distinguishable(self, covered_area):
    # Area up to the minimum borders is not covered
    covered_area = enlarge_mask(self.inner_size(), covered_area, False)

    for i, ows in reversed(list(enumerate(self.__objects_with_shifts))):
      if not ows.is_distinguishable(enlarge_mask(ows.bound_size(), covered_area, True)):
        return False

      covered_area = area_union([
        covered_area,
        ows.covered_area()
      ])

    return True

class Shift:
  def __init__(self, name, left_top):
    self.name = name

    assert_is_type(left_top, "(Left,Top)", "Shift " + name)

    self.__left_top = left_top

  def get(self):
    return np.array([
      self.__left_top.get_by_type("Left"),
      self.__left_top.get_by_type("Top")
    ])

  def type(self):
    return "Shift"

  def n_possibilities(self):
    return self.__left_top.n_possibilities()

  def max_entropy(self):
    return self.__left_top.max_entropy()

  def choices(self):
    return map(np.array, self.__left_top.choices())

def read_object_dict(yaml_objects, base_path):
  objects = {}

  def parse_object(name, yaml_obj, call_stack):
    subobjects = []

    is_union = len(yaml_obj) != 1

    if type(yaml_obj) != dict:
      raise ValueError("Object " + name + " must be a dictionary specifying types and "
        "their instantiations.")

    for k, v in yaml_obj.items():
      if is_union:
        curr_name = name + "." + k
      else:
        curr_name = name

      def expect_type(curr_type_name, expected_type, message):
        if type(v) != expected_type:
          raise ValueError("'" + curr_type_name + "' object " + curr_name + " " + message)

      def expect_aggregate(type_name, expected_list_elements, class_constructor,
          element_function=lambda i, po: po):
        expect_type(type_name, list, "expects a list of "
            + expected_list_elements + " as input. Given input was not a list.")

        elements = []
        for i, o in enumerate(v):
          po = parse_object(curr_name + "." + str(i), o, call_stack)

          elements.append(element_function(i, po))
        subobjects.append(class_constructor(curr_name, elements))

      if k == "object":
        subobjects.append(get_object(v, call_stack))
      elif k == "overlay":
        def default_to_no_shift(i, po):
          if i > 0 and po.type() == "Image":
            # add default shift if parsed object is just an Image
            shiftname = po.name + ".shift"
            return Union(po.name, [po, Shift(shiftname,
              Union(shiftname + ".coordinates", [
                Left(shiftname + ".coordinates.left", 0),
                Top(shiftname + ".coordinates.top", 0)
              ])
            )])
          return po

        expect_aggregate("Overlay", "a base image and any number of potentially "
          "shifted image for overlaying",
          Overlay, default_to_no_shift)
      elif k == "choice":
        expect_aggregate("Choice", "choices", Choice)
      elif k == "proportional_choice":
        expect_aggregate("ProportionalChoice", "choices", ProportionalChoice)
      elif k == "image":
        # leaf node
        # expects a single string as input
        subobjects.append(BaseImage(curr_name, v, base_path))
      elif k == "shift":
        left_top = parse_object(curr_name + "coordinates", v, call_stack)

        subobjects.append(Shift(curr_name, left_top))
      elif k == "left":
        subobjects.append(Left(curr_name, v))
      elif k == "top":
        subobjects.append(Top(curr_name, v))
      elif k == "union":
        # explicit union
        expect_aggregate("Union", "union elements", Union)
      elif k == "combined_shift":
        expect_aggregate("CombinedShift", "additive shifts", CombinedShift)
      else:
        raise ValueError("Unknown object kind " + k + " in " + name + "!")

    if is_union:
      return Union(name, subobjects)
    else:
      return subobjects[0]

  def get_object(obj_name, call_stack = []):
    if obj_name in objects:
      return objects[obj_name]

    if obj_name in call_stack:
      raise ValueError("Detected recursive dependency while parsing object "
        + obj_name + "!")

    return parse_object(obj_name, yaml_objects[obj_name], call_stack)

  for k in yaml_objects.keys():
    objects[k] = get_object(k)

  return objects

def read_scene_yaml(scene_description_path):
  path = Path(scene_description_path).expanduser()
  with path.open() as f:
    input_dict = yaml.safe_load(f)

  return read_object_dict(input_dict, path.parent)

class SceneGenerator:
  distinction_failure_message = None

  def __init__(self, scene_description_path, root="scene"):
    """
    Args:
      root: Object within the scene description to generate.
    """
    objects = read_scene_yaml(scene_description_path)

    self.__generator = objects[root]

    if self.__generator.type() != "Image":
      raise TypeError("Type of root object should be 'Image', found "
        + self.__generator.type() + " instead.")

  def generate(self):
    return self.__generator.get()

  def n_possibilities(self):
    """
    Prints the number of different choices this generator can make.

    This is not necessarily the same as the number of possible outcomes since different
    choices might lead to the same image.
    """
    return self.__generator.n_possibilities()

  def max_entropy(self):
    """
    Returns the maximum entropy this generator might have in case all choices lead to
    different results.
    """
    return self.__generator.max_entropy()

  def trace_distinction_failure(self):
    return SceneGenerator.distinction_failure_message

  def fully_distinguishable(self):
    """
    If this method returns true all images generated by the SceneGenerator
    are different and the estimated max_entropy is also the entropy.

    The converse it not true: If this returns false, the entropy might still be maximal.

    If this is False, trace_distinction_failure() will return an error explanation.
    """
    # Idea: Two areas are computed:
    # - covered area:
    #   - At least partially opaque pixels for images
    #   - Union of covered areas for choices or overlays
    # - identifiable area:
    #   - Fully opaque pixels which are not priorly covered for images
    #   - Intersection of identifiable areas for choices
    #   - Maximum identifiable area of any single element of overlays
    #
    # Distinguishability of elements are defined according to the possible value
    # their identifiable sets can take. Note that all these definitions are bounded
    # by the number of leaf images in the scene since the overlay definition of
    # identifiable areas chooses a single path.
    # - Image: Distinguishable if the identifiable area is not empty.
    # - Choice: Distinguishable if all elements take different values on the identifiable
    #   set of the choice.
    # - Overlay: Distinguishable if all elements are distinguishable.
    #
    # In total this process requires 4 functions:
    #  1. covered_area()
    #  2. identifiable_area(covered_area)
    #  3. identifying_values(identifying_area)
    #  4. is_distinguishable(covered_area)
    #
    # Areas will be represented as 2-d numpy arrays with top left corner aligned with
    # the currently considered image.
    SceneGenerator.distinction_failure_message = None
    return self.__generator.is_distinguishable(np.zeros((0,0), dtype=np.bool))

def run_test():
  pass

if __name__=="__main__":
  run_tests()
