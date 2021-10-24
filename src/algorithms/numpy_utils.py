import numpy as np

import tempfile
from os.path import join as join_paths

def all_over(arr, axis=0):
  """
  Returns whether along the given axis all elements are true.
  """
  dims = tuple(range(len(arr.shape)))
  return arr.all(axis=dims[:axis] + dims[axis+1:])

def any_over(arr, axis=0):
  """
  Returns whether along the given axis any elements are true.
  """
  dims = tuple(range(len(arr.shape)))
  return arr.any(axis=dims[:axis] + dims[axis+1:])

def mmap_constructor_on(filename):
  """
  Returns a function that constructs an mmap on the given file in write mode.
  """
  def mmap_constructor(shape, dtype=np.ubyte):
    return np.memmap(filename, mode='w+', shape=shape, dtype=dtype)
  return mmap_constructor

def tmp_mmap(shape, dtype=np.ubyte, order='C'):
  """
  Creates a numpy array within a temporary file for the purpose of usage under
  a limited amount of working memory.

  Args:
    order: Whether to store multi-dimensional data in row-major (C-style) or
      column-major (Fortran-style) order in memory.
  """
  with tempfile.NamedTemporaryFile() as ntf:
    return np.memmap(ntf.name, mode='w+', shape=shape, order=order, dtype=dtype)

def mmap_copy(array, filename=None):
  """
  Create a copy of the array in an mmap. By default uses a temporary file. The
  memmap will always be opened in write mode.
  """
  if filename is None:
    mm = tmp_mmap(array.shape, array.dtype)
  else:
    mm = np.memmap(filename, mode='w+', shape=array.shape, dtype=array.dtype)

  mm[:] = array

  return mm

def object_view(arr):
  """
  Returns a view of the given array where each subarray is interpreted as an
  object.
  """
  return np.ndarray(arr.shape[0], dtype=[('',arr.dtype)]*np.prod(arr.shape[1:]), buffer=arr)

def lexsort_in_place(arr):
  """
  Sorts the array lexicographically in-place.
  """
  # create a view of the data with object elements of the original array
  view = object_view(arr)
  # sort in-place
  view.sort()

  return arr

def to_2d(x):
  """
  Returns a 2d representation of the object.
  """
  shape_new = (len(x),np.prod((1,) + x.shape[1:]))
  return x.reshape(shape_new)

def combine_data(arrays, empty_constructor=np.empty, axis=-1, dtype=None):
  """
  Combines any number of arrays along an existing axis into another array.

  If multiple axises are given as argument then the arrays will be iteratively
  reduced along these axises starting from the first axis at the innermost
  array level. Each reduction will reduce the array list structure by 1 level.

  Data type will be that of the first element if not provided explictely.
  """
  if len(arrays) == 0:
    raise ValueError("Cannot combine empty list of arrays.")

  if not type(axis) == tuple or type(axis) == list:
    if type(axis) == int:
      axis = (axis,)
    else:
      raise TypeError("axis must be either an int or a tuple or list")

  # reverse axis to specify order starting at outermost level
  axis = tuple(reversed(axis))

  def shape_from_triple(triple):
    return triple[0] + triple[1] + triple[2]

  def make_shape(shapes, axis):
    """
    Gather the shape that the final object should have.
    """
    l = 0

    shape_determined = False

    for s in shapes:
      l += s[axis]

      this_shape_start = s[:axis]
      if axis == -1:
        this_shape_end = ()
      else:
        this_shape_end = s[axis+1:]
      if not shape_determined:
        shape_start = this_shape_start
        shape_end = this_shape_end
        shape_determined = True
      else:
        if shape_start != this_shape_start or shape_end != this_shape_end:
          first_shape = shape_from_triple((shape_start,(-1,),shape_end))
          second_shape = shape_from_triple((this_shape_start,(-1,),this_shape_end))
          raise ValueError("Can't stack shapes " + str(first_shape)
            + " and " + str(second_shape) + ".")

    return shape_start, (l,), shape_end

  def make_shape_stack(arrays, axis):
    """
    Gets ((outer_start_shape, outer_l, outer_end_shape), [inner_tuple] or [None], dtype)
    dtype is not included again in inner tuple
    """
    if len(axis) == 1:
      return (
        make_shape(map(lambda x: x.shape, arrays), axis[0]),
        [None]*len(arrays), arrays[0].dtype
      )

    shapes = []
    inner_tuples = []

    dtype = None

    for a in arrays:
      inner_tuple = make_shape_stack(a, axis[1:])
      if dtype is None:
        dtype = inner_tuple[2]
      inner_tuples.append(inner_tuple[:2])
      shapes.append(shape_from_triple(inner_tuple[0]))

    return make_shape(shapes, axis[0]), inner_tuples, dtype

  shape_stack = make_shape_stack(arrays, axis)

  # create combined
  combined = empty_constructor(shape_from_triple(shape_stack[0]), dtype=shape_stack[-1])
  
  def make_slices(shape):
    return tuple(map(slice, shape))

  def copy_over_stack(shape_stack, arrays, target):
    """
    Copy over data from arrays into target according to shape_stack.
    """
    if shape_stack is None:
      # base case: Copy over
      target[:] = arrays
    else:
      triple, inner_tuples = shape_stack

      shape_start, _, shape_end = triple

      slices_start = make_slices(shape_start)
      slices_end = make_slices(shape_end)

      start = 0
      for a, inner_stack in zip(arrays, inner_tuples):
        if inner_stack is None:
          target_shape = a.shape
        else:
          target_shape = shape_from_triple(inner_stack[0])

        end = start + target_shape[len(shape_start)]
        copy_over_stack(
          inner_stack,
          a,
          target[slices_start + (slice(start, end),) + slices_end]
        )
        start = end

  copy_over_stack(shape_stack[:2], arrays, combined)

  return combined

def separate_data(combined, shapes):
  """
  Args:
    shapes: List of shapes that the result should have. Do not include data
      length; first dimension will be inferred from combined.
  """
  constraints = []

  curr = 0

  for s in shapes:
    constr = (curr, curr + np.prod((1,) + s), combined.shape[:1] + s)
    constraints.append(constr)
    curr = constr[1]

  if curr != combined.shape[1]:
    raise ValueError("Could not match shape " + str(combined.shape) +
      " with shapes of total size " + str(curr))

  return tuple(map(
    lambda ses: combined[:,ses[0]:ses[1]].reshape(ses[2]),
    constraints))

def stagnants(x):
  """
  Calculates whether each element is the same as the element before it.
  """
  same = np.empty(len(x),dtype=np.bool)
  same[0] = False
  same[1:] = all_over(x[:-1] == x[1:])

  return same

def does_repeat(x):
  """
  Calculates whether each element is the same as the element after it.
  """
  same = np.empty(len(x),dtype=np.bool)
  same[-1] = False
  same[:-1] = all_over(x[:-1] == x[1:])

  return same

def bytes_for_size(size):
  """
  Return number of bytes necessary to cover the given size.
  """
  return int(np.ceil(np.log2(size) / 8.))

def index_array(size,rep_size=None):
  """
  Returns a 2d index array of type uint8 for making sure that indexes are in
  range of whatever type is being used in argsort.

  Args:
    rep_size: Amount of bytes to use for representation. This will lead to
      repeated indices if less than bytes_for_size(size).
  """
  if rep_size is None:
    rep_size = bytes_for_size(size)

  array = np.empty((size,rep_size), dtype=np.uint8)

  for i in range(rep_size):
    array[:,-(i+1)] = np.repeat(np.arange((size-1) // 256**i + 1), 256**i)[:size]

  return array

def collapse_index_array(indices):
  """
  Converts the bytewise index array to uint64 representation.
  """
  return indices.dot(2**(np.arange(indices.shape[1]-1,-1,-1)*8))

def mmap_argsort(data,mmap_on=None):
  """
  Sorts the data within a new mmap that also includes indices, then returns
  views of both the indices and the elements. Note that indices have to be
  converted to integers before usage as indices.

  This is mainly useful in cases where argsort is too slow because data is
  within an mmap.

  Args:
    mmap_on: If set to a filename, will construct the combined mmap in that
      filename and also return the combined array as last object. If None,
      a temporary mmap will be used and no combined object be returned.
  """
  if mmap_on is None:
    mmap_maker = tmp_mmap
  else:
    mmap_maker = mmap_constructor_on(mmap_on)

  combined = combine_data([to_2d(data), index_array(len(data))], mmap_maker)
  lexsort_in_place(combined)
  
  data, indices = separate_data(combined, [data.shape[1:], (bytes_for_size(len(data)),)])

  indices = collapse_index_array(indices)

  if mmap_on is None:
    return data, indices
  else:
    return data, indices + (combined,)

def changes(x):
  """
  Calculates whether each element is different from the element before it.
  """
  different = np.empty(len(x),dtype=np.bool)
  different[0] = True
  different[1:] = any_over(x[:-1] != x[1:])

  return different

def is_last(x):
  """
  Calculates whether each element is different from the element after it.
  """
  different = np.empty(len(x),dtype=np.bool)
  different[-1] = True
  different[:-1] = any_over(x[:-1] != x[1:])

  return different

def is_different(x):
  """
  Calculates whether is different from both adjacent elements.
  """
  return np.logical_and(is_last(x), changes(x))

def is_same(x):
  """
  Calculates whether is different from both adjacent elements.
  """
  return np.logical_or(stagnants(x), does_repeat(x))
