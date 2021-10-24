def get_from_multiple(listlikes, index, return_list=False):
  """
  Returns an element from a series of listlikes by spilling over to the next
  listlike if the index is not in the first.
  """
  for l in listlikes:
    if len(l) > index:
      if return_list:
        return l[index], l
      return l[index]
    else:
      index -= len(l)

  raise IndexError("Index isn't smaller than sum of lengths.")
