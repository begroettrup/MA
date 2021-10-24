def add_if_missing(dict_, key, value):
  """
  Adds a key-value pair to the given dictionary if it does not contain an entry
  for the given key.
  """
  if key not in dict_:
    dict_[key] = value
