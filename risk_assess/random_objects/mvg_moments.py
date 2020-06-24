import casadi
import numpy as np
from .utils import *

def mvg_mgf(t, mu, sigma):
    """Multivariate Gaussian Moment Generating Function.

    Args:
        t (1D array): 
        mu (1D array): 
        sigma (2D array): 
    """
    return casadi.exp(mu.T @ t + 0.5 * t.T @ sigma @ t)

def mvg_moment_array_functions(max_order, dimension, symbolic_vars = None):
    """ 
    Args:
        max_order ([type]): [description]
        dimension ([type]): [description]

    Returns:
        [type]: Resulting functions have signature (t, mu, sigma) where 
                t and mu are vectors, and sigma is the covariance matrix.
    """
    if symbolic_vars == None:
        # Declare variables.
        t = casadi.MX.sym('t', dimension, 1)
        mu = casadi.MX.sym("mu", dimension, 1)
        sigma = casadi.MX.sym("sigma", dimension, dimension)
    else:
        t, mu, sigma = symbolic_vars

    # This dictionary stores our resulting functions.
    moment_array_functions = dict()
    moment_array_syms = dict()

    # Start auto-differentiating the multivariate Gaussian MGF>
    foo = mvg_mgf(t, mu, sigma)
    for i in range(max_order):
        order = i+1
        foo = casadi.jacobian(foo, t) #TODO: current matrix output form doesn't quite make sense
        fun = casadi.Function('moment_array_order' + str(order), [t, mu, sigma], [foo])
        moment_array_syms[order] = foo
        moment_array_functions[order] = fun
    return moment_array_functions, moment_array_syms


def generate_mvg_moment_functions(max_order, dimension):
    """[summary]

    Args:
        max_order ([type]): [description]
        dimension ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Declare variables.
    t = casadi.MX.sym('t', dimension, 1)
    mu = casadi.MX.sym("mu", dimension, 1)
    sigma = casadi.MX.sym("sigma", dimension, dimension)

    _, array_syms = mvg_moment_array_functions(max_order, dimension, (t, mu, sigma))

    moment_functions = dict()
    for order in range(1, max_order + 1):
        possible_multi_idxs = constant_sum_tuples(dimension, order)
        for multi_idx in possible_multi_idxs:
            # Go from multi to array idx.
            tensor_idx = multi_to_tensor_idx(multi_idx, dimension)
            array_idx = tensor_to_array_idx(tensor_idx, dimension)

            # Generate the function.
            symbolic = array_syms[order][array_idx]
            fun_name = "moment" + "".join(map(str,multi_idx))
            moment_functions[multi_idx] = casadi.Function(fun_name, [t, mu, sigma], [symbolic])
    return moment_functions

def mvg_moments(mu, sigma, max_order):
    """ Compute all moments up to the specified order. Returns moments
        as a dictionary with moment multi-index as the key.

    Args:
        mu ([type]): [description]
        sigma ([type]): [description]
        max_order ([type]): [description]
    """
    # Compute the moment tensors.
    dimension = mu.size
    t = np.zeros(dimension)
    if sigma.shape != (dimension, dimension):
        raise Exception("Incompatible dimensions for mu and sigma.")

    moment_funcs = generate_mvg_moment_functions(max_order, dimension)

    # Populate moment_values with moments up to "max_order".
    moment_values = dict()
    for order in range(1, max_order+1):
        # All possible moment multi-indices of the current order.
        possible_multi_idxs = constant_sum_tuples(dimension, order)
        for multi_idx in possible_multi_idxs:
            moment_values[multi_idx] = moment_funcs[multi_idx](t, mu, sigma)
    return moment_values

class MVG:
    # Dictionary mapping multi-indicies to functions
    # that compute moments with signatures (t, mu, sigma)
    moment_functions = dict()
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self._dimension = mu.size

        # Check that the covariance matrix size is consistent.
        assert self._sigma.shape == (self._dimension, self._dimension)

        # Use self._t to compute moments.
        self._t = np.zeros(self._dimension)
    
    @classmethod
    def compile_moment_functions_up_to(cls, dimension, max_order):
        """ Compile all moment functions up to a certain order.

        Args:
            max_order (int): maximum order to compile functions up to.
        """
        new_funcs = generate_mvg_moment_functions(max_order, dimension)
        cls.moment_functions.update(new_funcs)
        return
    
    def compute_moment(self, multi_idx):
        if multi_idx not in MVG.moment_functions.keys():
            raise Exception("A moment function was not compiled to compute the desired moment.")
        fun = MVG.moment_functions[multi_idx]
        return float(fun(self._t, self._mu, self._sigma))
    
    def moments_up_to(self, max_order):
        """ Compute moments of order up to "max_order".

        Args:
            max_order (int): maximum order of moments to compute.
        """
        # Populate moment_values with moments up to "max_order".
        moment_values = dict()
        for order in range(1, max_order+1):
            # All possible moment multi-indices of the current order.
            possible_multi_idxs = constant_sum_tuples(self._dimension, order)
            for multi_idx in possible_multi_idxs:
                moment_values[multi_idx] = MVG.moment_functions[multi_idx](self._t, self._mu, self._sigma)
        return moment_values