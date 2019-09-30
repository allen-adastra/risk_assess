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
import math

class StochasticVerificationFunction(object):
    def __init__(self, p, stochastic_model):
        #p: an anonymous function in state
        #stochastic_model:
        self.p = p
        self.stochastic_model = stochastic_model
        self.coef_funcs = {}
        self.monoms = {}

    def compile_moment_functions_multinomial(self):
        """
        Compile functions to evaluate E[p(x,y)] and E[p(x,y)^2] by leveraging
        the multinomial expansion.
        """
        final_state = self.stochastic_model.get_final_state()
        # Symbolic first moment.
        p_first_moment = self.p(final_state[0], final_state[1])

        # Variables that are input at runtime.
        input_vars = self.stochastic_model.get_input_vars()

        # Compile the coefficients and monomials.
        self.coef_funcs[1] = [ufuncify(input_vars, coef) for coef in p_first_moment.coeffs()]
        self.monoms[1] = p_first_moment.monoms()

        n_monoms = len(self.monoms[1])
        # TODO: clear up this confusion
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

    def compute_prob_bound_multimonial(self, input_variables):
        coef_data = self.stochastic_model.listify_input_vars(input_variables) # TODO: how to best do this?
        p_first_coefs = [coef_func(*coef_data) for coef_func in self.coef_funcs[1]]
        start_compute_rv_moments = time.time()
        p_first_monomoments, p_second_monomoments = self.compute_rv_moments()
        print("Time to compute rv moments: " + str(time.time() - start_compute_rv_moments))
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
        prob_bound = self.chebyshev_bound(p_first_moment, p_second_moment)
        print("prob bound is: " + str(prob_bound))
        
    def chebyshev_bound(self, first_moment, second_moment):
        #bound the probability that p<=0
        if first_moment<=0:
            return None
        else:
            variance = second_moment - first_moment**2
            return variance/(variance + first_moment**2)
    
    def set_random_vector(self, random_vector):
        self.random_vector = random_vector

    def monte_carlo_result(self, input_vars, n_samples, dt):
        fails = 0
        for i in range(n_samples):
            sampled_accels = self.random_vector.sample()
            x = input_vars.x0
            y = input_vars.y0
            v = input_vars.v0
            for j in range(len(sampled_accels)):
                x += dt * v * math.cos(input_vars.thetas[j])
                y += dt * v * math.sin(input_vars.thetas[j])
                v += dt * sampled_accels[j]
            final_p = self.p(x, y)
            if final_p <= 0:
                fails+=1
        return fails/n_samples

    def compute_rv_moments(self):
        n_vars = len(self.monoms[1][0])
        monom_one_max_moments = []
        monom_two_max_moments = []
        for i in range(n_vars):
            monom_one_max_moments.append(max([x[i] for x in self.monoms[1]]))
            monom_two_max_moments.append(max([y[i] for y in self.monoms[2]]))
        # For the ith random variable, determine the maximum moment we would need for it.
        max_moments = [max(monom_one_max_moments[i], monom_two_max_moments[i]) for i in range(len(monom_one_max_moments))]
        moments = self.random_vector.compute_vector_moments(max_moments)

        # For each tuple in self.monoms[1] or self.monoms[2], the ith entry corresponds to the degree of that particular variable
        # in the monomial.
        mono1_moments = [np.prod([moments[i][mono[i]] for i in range(len(mono))]) for mono in self.monoms[1]]
        mono2_moments = [np.prod([moments[i][mono[i]] for i in range(len(mono))]) for mono in self.monoms[2]]
        return mono1_moments, mono2_moments