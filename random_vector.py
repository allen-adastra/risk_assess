from itertools import accumulate
import numpy as np
from scipy.special import hyp1f1

"""cbeta is the beta distribution multiplied by a constant. When c = 1, this
is just a normal beta random variable."""
class cBetaRandomVariable(object):
    def __init__(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def compute_moments(self, order):
        #Compute all beta moments up to the given order
        #the returned list indices should match the moment orders
        #e.g. the return[i] should be the ith beta moment
        fs = map(lambda r: (self.alpha + r)/(self.alpha + self.beta + r), range(order))
        beta = [1] + list(accumulate(fs, lambda prev,n: prev*n))
        cbeta = [beta[i]*self.c**i for i in range(len(beta))]
        return cbeta

    def sample(self):
        return self.c * np.random.beta(self.alpha, self.beta)

    def compute_characteristic_function(self, t):
        """
        The characteristic function of the beta distribution is Kummer's confluent hypergeometric function which 
        is implemented by scipy.special.hyp1f1. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html
        """
        return hyp1f1(self.alpha, self.beta, self.c * t)

    def compute_cosine_moment(self, n):
        """

        """

class RandomVector(object):
    def __init__(self, random_variables):
        # random_variables: list of random variables
        self.random_variables = random_variables

    def compute_vector_moments(self, n_moments):
        # n_moments is a list of numbers of the maximum moment order to be computed for each random variable
        all_moments = [] #list of list
        for i in range(len(self.random_variables)):
            all_moments.append(self.random_variables[i].compute_moments(n_moments[i]))
        return all_moments

    def sample(self):
        return [var.sample() for var in self.random_variables]

    def sum_characteristic_function(self, c, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """