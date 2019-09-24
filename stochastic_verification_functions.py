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

class StochasticVerificationFunction(object):
    def __init__(self, p, dynamics, n_steps, dt):
        #p: an anonymous function in state
        #dynamics: something something dynamics class, for now just CarKinematicsNoisyAccel
        self.p = p
        self.dynamics = dynamics(n_steps, dt)
        self.coef_funcs = {}
        self.monoms = {}

    def compile_moment_functions(self):
        """
        Compile functions to evaluate E[p(x,y)] and E[p(x,y)^2].
        """
        final_state = self.dynamics.get_final_state_vec()
        x_final = final_state[0]
        y_final = final_state[1]
        p_first_moment = self.p(x_final, y_final)
        p_second_moment = self.p(x_final, y_final)**2
        input_vars = list(self.dynamics.thetas) + [self.dynamics.x0, self.dynamics.y0, self.dynamics.v0]

        self.coef_funcs[1] = [ufuncify(input_vars,coef) for coef in p_first_moment.coeffs()]
        self.coef_funcs[2] = [ufuncify(input_vars,coef) for coef in p_second_moment.coeffs()]
        self.monoms[1] = p_first_moment.monoms()
        self.monoms[2] = p_second_moment.monoms()

    def compile_moment_functions_multinomial(self):
        """
        Compile functions to evaluate E[p(x,y)] and E[p(x,y)^2] by leveraging
        the multinomial expansion.
        """
        final_state = self.dynamics.get_final_state_vec()

        # The zeroth moment is always just one.
        self.p_moments[0] = 1
        # Symbolic first moment.
        p_first_moment = self.p(final_state[0], final_state[1])
        # Variables that are input at runtime.
        input_vars = list(self.dynamics.thetas) + [self.dynamics.x0, self.dynamics.y0, self.dynamics.v0]

        # Compile the coefficients and monomials.
        self.coef_funcs[1] = [ufuncify(input_vars, coef) for coef in p_first_moment.coeffs()]
        self.monoms[1] = p_first_moment.monoms()
        n_monoms = len(self.monoms[1])

        # TODO(allen): what the fuck?
        comb_twos = list(itertools.combinations(range(n_monoms), 1))
        comb_ones = list(itertools.combinations(range(n_monoms), 2))
        monom_twos = len(comb_twos)*[None]
        monom_ones = len(comb_ones)*[None]

        for i in range(len(comb_twos)):
            new_array = n_monoms*[0]
            new_array[comb_twos[i][0]] = 2
            monom_twos[i] = tuple(new_array)

        for i in range(len(comb_ones)):
            new_array = n_monoms*[0]
            for n in comb_ones[i]:
                new_array[n] = 1
            monom_ones[i] = tuple(new_array)

        #variables are the terms of p!
        p2_var_monoms = monom_twos + monom_ones

        #Figure out self.monoms[2]
        p2_moment_monoms = []
        for mono in p2_var_monoms:
            if 1 in mono:
                idx = [i for i,x in enumerate(mono) if x == 1]
                assert(len(idx) == 2)
                new_mono = tuple(map(operator.add,self.monoms[1][idx[0]],self.monoms[1][idx[1]]))
            elif 2 in mono:
                new_mono = tuple(map(operator.add,self.monoms[1][mono.index(2)],self.monoms[1][mono.index(2)]))
            else:
                raise Exception("There should be a 1 or 2 in here...")
            p2_moment_monoms.append(new_mono)
        self.monoms[2] = p2_moment_monoms
        self.p2_var_monoms = p2_var_monoms

    def compute_prob_bound_multimonial(self):
        coef_data = list(self.thetas) + [self.x0, self.y0, self.v0]
        p_first_coefs = [coef_func(*coef_data) for coef_func in self.coef_funcs[1]]
        p_first_monomoments, p_second_monomoments = self.compute_rv_moments()
        p_second_coefs = len(self.p2_var_monoms)*[0]
        for count, mono in enumerate(self.p2_var_monoms):
            if 1 in mono:
                idx = [i for i,x in enumerate(mono) if x==1]
                coefs = 2*p_first_coefs[idx[0]]*p_first_coefs[idx[1]]
            elif 2 in mono:
                i = mono.index(2)
                coefs = p_first_coefs[i]**2
            else:
                raise Exception("There should be a 1 or 2 in here...")
            p_second_coefs[count] = coefs
        p_first_moment = np.dot(p_first_coefs, p_first_monomoments)
        p_second_moment = np.dot(p_second_coefs, p_second_monomoments)
        print("first moment is: " + str(p_first_moment))
        print("second moment is: " + str(p_second_moment))
        prob_bound = self.chebyshev_bound(p_first_moment, p_second_moment)
        print("prob bound is: " + str(prob_bound))

    def compute_p_second_coefs(self, p_first_coefs):
        pass

    def chebyshev_bound(self, first_moment, second_moment):
        #bound the probability that p<=0
        if first_moment<=0:
            return None
        else:
            variance = second_moment - first_moment**2
            return variance/(variance + first_moment**2)


    def set_problem_data(self, uaccel_seq, random_vector, theta_seq, x0, y0, v0):
        self.random_vector = random_vector
        self.random_vector.set_cvals(uaccel_seq)
        self.thetas = theta_seq
        self.x0 = x0
        self.y0 = y0
        self.v0 = v0

    def compute_rv_moments(self):
        n_vars = len(self.monoms[1][0])
        x_max_moments = []
        y_max_moments = []
        for i in range(n_vars):
            x_max_moments.append(max([x[i] for x in self.monoms[1]]))
            y_max_moments.append(max([y[i] for y in self.monoms[2]]))
        max_moments = [max(x_max_moments[i], y_max_moments[i]) for i in range(len(x_max_moments))]
        moments = self.random_vector.compute_vector_moments(max_moments)

        #for each mono, the ith entry corresponds to the ith rv and the moment we want....
        mono1_moments = [reduce(lambda a,b: a*b, map(lambda i:moments[i][mono[i]] , range(len(mono)))) for mono in self.monoms[1]]
        mono2_moments = [reduce(lambda a,b: a*b, map(lambda i:moments[i][mono[i]] , range(len(mono)))) for mono in self.monoms[2]]

        return mono1_moments, mono2_moments

    def compute_prob_bound(self):
        coef_data = list(self.thetas) + [self.x0, self.y0, self.v0]

        p_first_coefs = [coef_func(*coef_data) for coef_func in self.coef_funcs[1]]
        p_second_coefs = [coef_func(*coef_data) for coef_func in self.coef_funcs[2]]
        p_first_monomoments, p_second_monomoments = self.compute_rv_moments()

        p_first_moment = np.dot(p_first_coefs, p_first_monomoments)
        p_second_moment = np.dot(p_second_coefs, p_second_monomoments)
        prob_bound = self.chebyshev_bound(p_first_moment, p_second_moment)
        print("prob bound is: " + str(prob_bound))



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
