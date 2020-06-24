import numpy as np
import cmath
from scipy.stats import norm
from scipy.special import hyp1f1
from itertools import accumulate

class RandomVariable(object):
    def __init__(self):
        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    """
    Virtual methods.
    """
    def compute_moment(self, order):
        raise NotImplementedError("Method compute_moments() is not implemented.")

    def sample(self):
        raise NotImplementedError("Method sample() is not implemented.")

    def compute_characteristic_function(self, t):
        raise NotImplementedError("Method compute_characteristic_function() is not implemented.")
    
    """
    Implemented methods.
    """
    def compute_moments(self, order):
        return [self.compute_moment(i) for i in range(order + 1)]

    def compute_variance(self):
        return self.compute_moment(2) - self.compute_moment(1)**2

class Normal(RandomVariable):
    def __init__(self, mean, std):
        self._mean = mean
        self._variance = std**2

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def clear_cache(self):
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            self._moment_values[order] = norm.moment(order, loc = self._mean, scale = self._variance**0.5)
        return self._moment_values[order]
    
    def compute_characteristic_function(self, t):
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(-0.5 * self._variance * t**2, self._mean * t))
        return self._char_fun_values[t]
    
    def sample(self, n_samps):
        return np.random.normal(self._mean, self._variance**0.5, size = (n_samps, 1))
    
    def scale(self, scale_factor):
        self._mean *= scale_factor
        self._variance *= (scale_factor**2)
        self.clear_cache()


# """cbeta is the beta distribution multiplied by a constant. When c = 1, this
# is just a normal beta random variable."""
class cBetaRandomVariable(RandomVariable):
    def __init__(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        return self.compute_moments(order)[order]

    def compute_moments(self, order):
        """
        Compute all beta moments up to the given order. The returned list indices should match the moment orders
        e.g. the return[i] should be the ith beta moment
        """
        fs = map(lambda r: (self.alpha + r)/(self.alpha + self.beta + r), range(order))
        beta = [1] + list(accumulate(fs, lambda prev,n: prev*n))
        cbeta = [beta[i]*self.c**i for i in range(len(beta))]
        return cbeta

    def sample(self, n_samps):
        return self.c * np.random.beta(self.alpha, self.beta, size = (n_samps, 1))

    def compute_characteristic_function(self, t):
        """
        The characteristic function of the beta distribution is Kummer's confluent hypergeometric function which 
        is implemented by scipy.special.hyp1f1. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html
        If Phi_X(t) is the characteristic function of X, then for any constant c, the characteristic function of cX is
        Phi_cX(t) = Phi_X(ct)
        """
        return hyp1f1(self.alpha, self.beta, self.c * t)