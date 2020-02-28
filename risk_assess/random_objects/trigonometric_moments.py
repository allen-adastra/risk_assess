from itertools import accumulate
import numpy as np
import math
import cmath
from scipy.special import comb
from scipy.stats import norm, ncx2, chi2
from risk_assess.random_objects.random_variables import RandomVariable

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

    def cos_sin(self):
        return CrossSumOfRVs(self.c, self.random_variables)

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
    
    def clear_cache(self):
        self._char_fun_values = {}
        self._moment_values = {}
        
    def compute_moment(self, order):
        # If there are no random variables, this is just a constant.
        if len(self.random_variables) == 0:
            return math.cos(self.c)**order

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
                    summation = sum([comb(2 * n, k, exact = True) * real_component[2 * (n - k)] for k in range(n)])
                    self._moment_values[order] = (1.0/(2.0**(2.0 * n))) * comb(2 * n, n, exact = True) + (1.0/(2.0**(2.0 * n - 1))) * summation
                elif order % 2 == 1:
                    summation = sum([comb(2 * n + 1, k, exact = True) * real_component[2 * n + 1 - 2 * k] for k in range(n + 1)])
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
        assert len(self.random_variables) > 0
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, t * self.c)) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]
    

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

    def clear_cache(self):
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        # If there are no random variables, this is just a constant.
        if len(self.random_variables) == 0:
            return math.sin(self.c)**order
        
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
                    summation = sum([((-1)**k) * comb(2 * n, k, exact = True) * real_component[2 * (n - k)] for k in range(n)])
                    self._moment_values[order] = (1.0/(2.0**(2.0 * n))) * comb(2 * n, n, exact = True) + (((-1)**n)/ (2**(2 * n - 1))) * summation
                elif order % 2 == 1:
                    summation = sum([((-1)**k) * comb(2 * n + 1, k, exact = True) * imaginary_component[2 * n + 1 - 2 * k] for k in range(n+1)])
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
        assert len(self.random_variables) > 0
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, t * self.c)) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
Let theta = c + x1 + x2 + ... + xn
This object is the random variable cos(theta) * sin(theta)
"""
class CrossSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        # If there are no random variables, this is just a constant
        if len(self.random_variables) == 0:
            return (math.cos(self.c) * math.sin(self.c))**order
            
        if order not in self._moment_values.keys():
            if order == 1:
                self._moment_values[order] = 0.5 * np.imag(self.compute_characteristic_function(2))
            else:
                raise Exception("Input order is not currently supported.")
        return self._moment_values[order]

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        assert len(self.random_variables) > 0
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] =  cmath.exp(complex(0, t * self.c)) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]