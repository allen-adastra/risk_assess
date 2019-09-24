import sympy as sp
import numpy as np
from functools import reduce
from itertools import accumulate
import time
from sympy.utilities.autowrap import ufuncify
from inspect import signature
import itertools
from itertools import permutations
import operator

class RandomVector(object):
    def __init__(self, n, distributions):
        #n: dimension of the random vector
        #distributions: dictionary with name as key and param list as val
        self.valid_distributions = {"beta", "cbeta"}
        if len(distributions)!=n:
            raise Exception("Der")
        elif len(list(filter(self.valid_distribution, distributions))) != n:
            raise Exception("Her")
        else:
            self.distributions = distributions
    
    def valid_distribution(self, distrib):
        if distrib[0] in self.valid_distributions:
            return True
        else:
            return False
    
    def compute_vector_moments(self, n_moments):
        #n_moments is a list of numbers of the maximum moment order to be computed
        all_moments = [] #list of list
        for i in range(len(self.distributions)):
            imoments = self.compute_moments(self.distributions[i], n_moments[i])
            all_moments.append(imoments)
        return all_moments
    
    def compute_moments(self, distr, order):
        if distr[0] == "beta":
            #Compute beta moments up to the given order
            #the returned list indices should match the moment orders
            #e.g. the return[i] should be the ith beta moment
            alpha = distr[1][0]
            beta = distr[1][1]
            fs = map(lambda r: (alpha + r)/(alpha + beta + r), range(order))
            return [1] + list(accumulate(fs, lambda prev,n: prev*n))
        
        elif distr[0] == "cbeta":
            alpha = distr[1][0]
            beta = distr[1][1]
            c = distr[1][2]
            fs = map(lambda r: (alpha + r)/(alpha + beta + r), range(order))
            beta = [1] + list(accumulate(fs, lambda prev,n: prev*n))
            cbeta = [beta[i]*c**i for i in range(len(beta))]
            return cbeta
        else:
            raise Exception("type not supported der")
        
    def set_cvals(self, cvals):
        #if cbeta distribution, this sets the values of c
        for i in range(len(cvals)):
            self.distributions[i][1][2] = cvals[i]

def compute_all_thetas(theta0, utheta):
    return theta0 + np.cumsum(utheta)

def compute_beta2_mono_moments(alpha, beta, monos):
    max_order_needed = max(max(monos))
    #assume iid beta moments
    beta2_moments = compute_beta2_moments(alpha,beta,max_order_needed)
    cross_moments = [reduce(lambda prev,ni: prev*beta2_moments[ni],[1] + list(tup)) for tup in monos]
    return cross_moments

def compute_beta_moments(alpha,beta,order):
    #Compute beta moments up to the given order
    #the returned list indices should match the moment orders
    #e.g. the return[i] should be the ith beta moment
    fs = map(lambda r: (alpha + r)/(alpha + beta + r), range(order))
    return [1] + list(accumulate(fs, lambda prev,n: prev*n))

def compute_beta2_moments(alpha, beta, n):
    beta_moments = compute_beta_moments(alpha, beta, n)
    return [beta_moments[i]*2**i for i in range(len(beta_moments))]

def chebyshev_bound(first_moment, second_moment):
    #bound the probability that p<=0
    if first_moment<=0:
        return None
    else:
        variance = second_moment - first_moment**2
        return variance/(variance + first_moment**2)