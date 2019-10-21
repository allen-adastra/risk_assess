from itertools import accumulate
import numpy as np
import math
import cmath
from scipy.special import hyp1f1, comb
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    
    def plot_histogram(self, n_samples, bins = None):
        samps = [self.sample() for i in range(n_samples)]
        if bins == None:
            bins = "auto"
        plt.hist(samps, bins = bins)
        plt.show()

class RandomVector(object):
    def __init__(self, random_variables):
        # random_variables: list of random variables
        self.random_variables = random_variables

    def compute_vector_moments(self, n_moments):
        # n_moments is a list of numbers of the maximum moment order to be computed for each random variable
        all_moments = len(n_moments) * [None] #list of lists
        for i in range(len(n_moments)):
            all_moments[i] = self.random_variables[i].compute_moments(n_moments[i])
        return all_moments

    def sample(self):
        return [var.sample() for var in self.random_variables]

    def dimension(self):
        return len(self.random_variables)

"""
Finite State Time-Homogenous Markov Chain. 
"""
class MarkovChain(object):
    def __init__(self, T, p0):
        """
        T: n x n transition matrix
        p0: n x 1 vector of transition probabilities
        """
        self.T = T
        self.p0 = p0

        # Dictionary mapping time step "i" to a vector of the
        # marginal probabilities at time "i"
        self.marginal_probabilities = {0 : p0}

    def get_marginal_vector(self, n_step):
        """
        Get the vector of marginal probabilities at time n_step.
        """
        assert isinstance(n_step, int) == True
        assert n_step >= 0
        if n_step not in self.marginal_probabilities.keys():
            # In this case, we don't have the desired marginal probability vector yet.

            # Identify the highest time step for which we have 
            # the vector of marginal probabilities
            max_i = max(self.marginal_probabilities.keys())

            # Propagate marginal probabilities forward in time.
            for i in range(max_i, n_step):
                # The vector of marginal probabilities at time t + 1 is the transition matrix multiplied
                # by the vector of marginal probabilities at time t
                self.marginal_probabilities[i + 1] = np.matmul(self.T, self.marginal_probabilities[i])
        return self.marginal_probabilities[n_step]

class MixtureModel(RandomVariable):
    def __init__(self, component_random_variables):
        """
        component_random_variables: tuples with (RandomVariable corresponding to a mode, probability of the mode)
        """
        self.component_random_variables = [rv for rv, prob in component_random_variables]
        self.component_probabilities = [prob for rv, prob in component_random_variables]
        sum_probs = sum(self.component_probabilities)
        if abs(sum_probs - 1) != 0:
            raise Exception("Input component probabilities must sum to 1, it sums to: " + str(sum_probs))
        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            moment = 0
            for rv, prob in zip(self.component_random_variables, self.component_probabilities):
                moment += prob * rv.compute_moment(order)
            self._moment_values[order] = moment
        return self._moment_values[order]

    def compute_characteristic_function(self, t):
        if t not in self._char_fun_values.keys():
            char_fun = 0
            for rv, prob in zip(self.component_random_variables, self.component_probabilities):
                char_fun += prob * rv.compute_characteristic_function(t)
            self._char_fun_values[t] = char_fun
        return self._char_fun_values[t]

    def sample(self):
        # Draw one sample from the multinomial distribution
        # np.random.multinomial(1, [0.2, 0.8]) for example will return either
        # array([0, 1]) or array([1, 0]) where the index of the 1 corresponds to
        # the variable chosen.
        mode_idx = list(np.random.multinomial(1, self.component_probabilities)).index(1)
        return self.component_random_variables[mode_idx].sample()

class Normal(RandomVariable):
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        self._variance = std**2

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}
    
    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            self._moment_values[order] = norm.moment(order, loc = self._mean, scale = self._std)
        return self._moment_values[order]
    
    def compute_characteristic_function(self, t):
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(-0.5 * self._variance * t**2, self._mean * t))
        return self._char_fun_values[t]
    
    def sample(self):
        return np.random.normal(self._mean, self._std)

class Constant(RandomVariable):
    def __init__(self, value):
        self.value = value

    def compute_moment(self, order):
        if order >= 0:
            return self.value**order
        else:
            raise Exception("Invalid order input")

    def sample(self):
        return self.value

    def compute_characteristic_function(self, t):
        return 1.0

"""cbeta is the beta distribution multiplied by a constant. When c = 1, this
is just a normal beta random variable."""
class cBetaRandomVariable(RandomVariable):
    def __init__(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moments(self, order):
        """
        Compute all beta moments up to the given order. The returned list indices should match the moment orders
        e.g. the return[i] should be the ith beta moment
        """
        fs = map(lambda r: (self.alpha + r)/(self.alpha + self.beta + r), range(order))
        beta = [1] + list(accumulate(fs, lambda prev,n: prev*n))
        cbeta = [beta[i]*self.c**i for i in range(len(beta))]
        return cbeta

    def sample(self):
        return self.c * np.random.beta(self.alpha, self.beta)

    def compute_characteristic_function(self, t):
        """
        The characteristic function of the beta distribution is Kummer's confluent hypergeometric function which 
        is implemented by scipy.special.hyp1f1. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html
        If Phi_X(t) is the characteristic function of X, then for any constant c, the characteristic function of cX is
        Phi_cX(t) = Phi_X(ct)
        """
        return hyp1f1(self.alpha, self.beta, self.c * t)

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
This object is the random variable c + x1 + x2 + ... + xn
"""
class SumOfRVs(object):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

    def cos_applied(self):
        return CosSumOfRVs(self.c, self.random_variables)

    def sin_applied(self):
        return SinSumOfRVs(self.c, self.random_variables)

    def add_rv(self, rv):
        self.random_variables.append(rv)

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
This object is the random variable cos(c + x1 + x2 + ... + xn)
"""
class CosSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}
        
    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            if order == 1:
                return np.real(self.compute_characteristic_function(1))
            elif order > 1:
                n = math.floor(order/2)
                # The ith element of real_component is the real component of CharacteristicFunction(i)
                char_fun_values = [self.compute_characteristic_function(i) for i in range(order + 1)]
                real_component = [np.real(val) for val in char_fun_values]
                # Different expressions depending on if the order is odd or even
                if order % 2 == 0:
                    summation = sum([comb(order, k) * real_component[2 * (n - k)] for k in range(n)])
                    self._moment_values[order] = (1.0/(2.0**(2.0 * n))) * comb(order, n) + (1.0/(2.0**(2.0 * n - 1))) * summation
                elif order % 2 == 1:
                    summation = sum([comb(2 * n + 1, k) * real_component[2 * n + 1 - 2 * k] for k in range(n + 1)])
                    self._moment_values[order] = (1.0/(4.0**n)) * summation
                else:
                    raise Exception("Input order mod 2 is neither 0 nor 1")
            else:
                raise Exception("Input order must be positive.")
        return self._moment_values[order]

    def compute_moments(self, order):
            return [self.compute_moment(i) for i in range(order + 1)]

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, t * self.c)) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]
    
    def add_rv(self, rv):
        self.random_variables.append(rv)
    
    def add_constant(self, c):
        self.c += c

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
This object is the random variable sin(c + x1 + x2 + ... + xn)
"""
class SinSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            if order == 1:
                return np.imag(self.compute_characteristic_function(1))
            elif order > 1:
                n = math.floor(order/2)
                # The ith element of real_component is the real component of CharacteristicFunction(i)
                char_fun_values = [self.compute_characteristic_function(i) for i in range(order + 1)]
                real_component = [np.real(val) for val in char_fun_values]
                imaginary_component = [np.imag(val) for val in char_fun_values]
                # Different expressions depending on if the order is odd or even
                if order % 2 == 0:
                    summation = sum([((-1)**k) * comb(order, k) * real_component[2 * (n - k)] for k in range(n)])
                    self._moment_values[order] = (1.0/(2.0**(2.0 * n))) * comb(order, n) + (((-1)**n)/ (2**(2 * n - 1))) * summation
                elif order % 2 == 1:
                    summation = sum([((-1)**k) * comb(order, k) * imaginary_component[2 * n + 1 - 2 * k] for k in range(n+1)])
                    self._moment_values[order] = (((-1)**n)/(4**n)) * summation
                else:
                    raise Exception("Input order mod 2 is neither 0 nor 1")
            else:
                raise Exception("Input order must be positive.")
        return self._moment_values[order]

    def compute_moments(self, order):
        return [self.compute_moment(i) for i in range(order + 1)]

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, 1) * t * self.c) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]

    def add_rv(self, rv):
        self.random_variables.append(rv)
    
    def add_constant(self, c):
        self.c += c

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
Let theta = c + x1 + x2 + ... + xn
This object is the random variable cos(theta) * sin(theta)
"""
class CrossSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables
        self.cos_theta = CosSumOfRVs(c, random_variables)
        self.sin_theta = SinSumOfRVs(c, random_variables)

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            if order == 1:
                self._moment_values[order] = 0.5 * np.imag(self.compute_characteristic_function(2))
            else:
                raise Exception("Input order is not currently supported.")
        return self._moment_values[order]

    def compute_covariance(self):
        return self.compute_moment(1) - self.cos_theta.compute_moment(1) * self.sin_theta.compute_moment(1)

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, 1) * t * self.c) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]