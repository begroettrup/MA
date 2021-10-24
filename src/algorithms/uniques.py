import algorithms.numpy_utils as npu

import numpy as np
import torch

def get_repeated(data, return_counts=False, return_locs=False):
  """
  Returns the elements that occur multiple times within the dataset.
  If any of the additional return flags are set instead returns the respective
  subset of the tuple (repeated, counts, locs)

  Args:
    return_counts: Also return how often each element occurred.
    return_locs: Also return locations where each repeated occurred as a list of
      arrays.
  """
  outs = np.unique(data,
    return_counts=True,
    return_inverse=return_locs,
    axis=0)
  elems = outs[0]
  counts = outs[-1]
  locs_multiples = np.flatnonzero(counts > 1)
  
  ret = (elems[locs_multiples],)

  if return_counts:
    ret = ret + (counts[locs_multiples],)

  if return_locs:
    locs = []
    inverses = outs[-2]

    for irep in locs_multiples:
      locs.append(np.flatnonzero(inverses == irep))
    ret = ret + (locs,)

  if len(ret) == 1:
    return ret[0]
  else:
    return ret

def get_counts(data):
  """
  Returns the count of different elments in the list-like of tensors.
  """
  inputs = torch.stack(list(map(lambda x: x[0], data)))
  _, counts = inputs.unique(dim=0, return_counts=True)
  return counts

def multiplicities(counts):
  """
  Given the (potentially unsorted) counts of elements, returns the
  multiplicities of the data, i.e. an array of how often each count occured in
  the data, starting from count 1.

  len(multiplicities(counts)) will be equal to counts[-1].
  """
  counts = np.array(counts)
  counts.sort()
  multiplicities = np.zeros(counts[-1],dtype=np.int)

  # locations of last elements of each identity
  change_indices = np.flatnonzero(npu.is_last(counts))
  
  # indices of the buckets counted by change_indices
  bucket_locations = counts[change_indices] - 1

  # calculate how large each bucket is
  # initialize as cumsum of bucket counts
  bucket_counts = change_indices
  del change_indices
  bucket_counts += 1
  # last element is equal to total number of elements
  # minus number of elements accounted for
  multiplicities[bucket_locations] = bucket_counts
  multiplicities[bucket_locations[1:]] -= bucket_counts[:-1]

  return multiplicities

def count_in_persistent(data):
  """
  Calculates counts in persistent memory.
  """
  data = npu.mmap_copy(data)
  npu.lexsort_in_place(data)
  return sorted_counts(data)

def sorted_counts(data):
  """
  Calculates the counts of elements in a sorted array.
  """
  # locations after changes
  # this is equal to the cumsum of counts
  counts = np.flatnonzero(npu.is_last(data)) + 1

  # undo cumsum by subtracting sum of all preceding elements
  counts[1:] -= counts[:-1]

  return counts

def sorted_unique_in_place(x):
  """
  Return an in-place view of the array that only contains unique elements.
  Assumes that the array is sorted
  """
  if len(x) == 0:
    return x

  return x[npu.changes(x)]

def unique_in_place(x):
  """
  Return an in-place view of the array that only contains unique elements.
  Sorts the array in-place.
  """
  return sorted_unique_in_place(npu.lexsort_in_place(x))

