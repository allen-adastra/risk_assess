from itertools import accumulate
import numpy as np

"""cbeta is the beta distribuiton multiplied by a constant. When c = 1, this
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