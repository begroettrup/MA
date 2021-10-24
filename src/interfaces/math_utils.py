def round_up_integer_divide(x,y):
  """
  For integer values or tensors of integers returns x / y rounded up.
  """
  return (x  - 1) // y + 1

def int_log(base, val):
  """
  Find the maximum exponent such that base**exp < val
  """
  exp = 1
  while base**(exp+1) < val:
    exp += 1

  return exp
