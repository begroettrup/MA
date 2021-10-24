import torch
import numpy as np
import scipy.special

def entropy_plug_in(counts):
  """
  Plug-in estimate of entropy ∑p_i ln p_i where p_i is the data frequency.
  """
  n = np.array(counts)
  N = n.sum()
  p = n / N
  return -(p*np.log(p)).sum()

def entropy_ansb(counts):
  """
  Asymptotic Nemenman-Shafee-Bialek estimator, see
  https://arxiv.org/pdf/physics/0108025.pdf
  https://www.mdpi.com/1099-4300/13/12/2013xml
  """
  # estimates
  # H = (Cγ - ln 2) + 2 ln N - ψ₀(Δ)
  # where Cγ is the Euler-Mascheroni constant, ψ₀ is the digamma function
  # and Δ is the number of coincidences, i.e. N - #(bins of size ≥1)
  n = np.array(counts)
  N = n.sum()
  Delta = N - len(n)
  return (np.euler_gamma - np.log(2)) + 2*np.log(N) - scipy.special.digamma(Delta)

def entropy_grassberger_baseline(n_data, precision=1e-6):
  """
  Returns the Grassberger entropy estimate assuming no collisions.
  """
  return entropy_grassberger(np.ones(n_data))

def entropy_grassberger(counts, precision=1e-6):
  """
  Calculates the entropy estimate from https://arxiv.org/pdf/physics/0307138.pdf
  (Grassberger, Entropy Estimates from Insufficient Samplings)

  Args:
    precision: Maximum error term.
  """
  def sum_term(n, l):
    nplus2l = n + 2*l
    return 1 / (nplus2l*(nplus2l+1))

  def error_term(aggregate, n, l):
    """
    Returns the absolute error term for Gn estimated by reduction to the
    reciprocals of squares:

    ∑_{i=k}^∞ 1/(n+2l)(n+2l+1) ≤ 1/4 ∑_{i=k}^∞ 1/(l+1)²
    = 1/4(π²/6 - ∑_{i=1}^{k-1}1/(l+1)²)
    using that ∑_{i=1}^∞ 1/l² = π²/6
    """
    # aggregate is sum of 1/(k+1)^2 with k=0,...,l-1
    if aggregate is None:
      aggregate = 0

    error = (np.pi**2 / 6 - aggregate)/4

    aggregate += 1/(l+1)**2
    return (error,aggregate)

  n = np.array(counts)

  phi = scipy.special.digamma(n) + (-1)**n * converging_series(
    n, sum_term,
    # actual is weighted average of Gn error
    # but the same upper bound is used for all Gnᵢ
    error_cap=precision,
    error_function=error_term,
    name="grassberger entropy")
  N = n.sum()

  correction = np.dot(n, phi) / N

  return np.log(N) - correction

def converging_series(x, f, min_iterations=0, max_iterations=None,
    error_cap=0., error_function=None,
    name="converging series"):
  """
  Calculate the sum of f(x,n) with n in [0,max_iterations-1].

  If no error function is given, assumes that the absolute value of f is
  strictly decreasing after min_iterations and terminates once
  |f(x,n)|<=error_cap .

  If an error function is given, terminates once the returned error is smaller
  or equal than error_cap.

  Args:
    max_iterations: maximum number of iterations. Smaller than 1 for infinite.
    name: name of the series for progress display
    error_function: Function of (aggregate, x, n) that returns a pair of
      (error,aggregate) where error is an estimate for the absolute error
      of the function and aggregate is an arbitrary aggregate value passed to
      the next call and initialized with None.
  """
  def make_error(dy, aggregate, n):
    """
    Returns error, aggregate.
    """
    if error_function is not None:
      return error_function(aggregate, x, n)
    else:
      return (dy.max(), aggregate)

  dy = f(x,0)
  y = dy
  error, aggregate = make_error(dy, None, 0)
  n = 1

  while n != max_iterations and error > error_cap:
    print("Calculating {}, error={:7.1e}".format(name, error), end="\r")
    dy = f(x,n)
    y += dy
    error, aggregate = make_error(dy, aggregate, n)
    n += 1

  print(" Calculated {}, error={:7.1e}".format(name, error))

  return y
