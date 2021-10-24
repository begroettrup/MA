import math

class DiscreteDistribution:
  """
  Models a discrete distribution.
  """
  def __init__(self, bucket_counts_per_size, n = None):
    """
    Create a discrete distribution with a total of n elements where there are
    bucket_counts_per_size[k] buckets with size k. 
    """
    if n is None:
      n = sum(bucket_counts_per_size)

    self.n = n
    self.bucket_counts_per_size = bucket_counts_per_size

  def entropy(self):
    """
    Return the entropy of this distribution.
    """
    return map(lambda x: math.log(x) * x, self.bucket_counts_per_size) / self.n

class BirthdayDistributions:
  """
  Models a distribution of discrete distributions produced by having a natural
  number associated with each element from a fixed size domain and sampling
  with linear probability of that number.
  """
  def __init__(self, domain_size, prob_dists=[(1.,DiscreteDistribution((1,)))]):
    """
    Create a distribution of distributions where prob_dists is a list of pairs
    (probability of distribution, distribution).

    Args:
      domain_size: Number of different buckets.
    """
    self.domain_size = domain_size
    self.prob_dists = prob_dists

  def expected_entropy(self):
    """
    Returns the expected entropy of the distribution of distributions.
    """
    return sum(map(lambda p, dist: p*dist.entropy(), self.prob_dists))

  def increase_elems(self):
    """
    Returns a distribution with the total sum per distribution increased by one.
    """
    prob_dists_new = []

    for p, dist in prob_dists:


    return BirthdayDistributions(self.domain_size, )
