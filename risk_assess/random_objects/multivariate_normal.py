import numpy as np
from .random_variables import Normal
from .mvg_moments import *

class MultivariateNormal(object):
    # Dictionary mapping multi-indicies, represented as tuples of ints, to
    # functions to compute the moments. The functions have the signature (t, mean, cov).
    # This is stored as a static variable s.t. the function does not need to be
    # repeatedly compiled.
    moment_functions = dict()
    compilations_complete = dict() # List of tuples dimension -> max_order to track what compilations have been done.

    def __init__(self, mean, covariance):
        """
        Args:
            mean (1D numpy array): mean vector
            covariance (n x n numpy array): covariance matrix
        """
        self._mean = mean.flatten()
        self._covariance = covariance
        self._dimension = mean.size

        # Check that the covariance matrix size is consistent.
        assert self._covariance.shape == (self._dimension, self._dimension)

        # t vector used in MGF to compute moments.
        self._t = np.zeros(self._dimension) 

    @property
    def mean(self):
        return self._mean
    
    @property
    def covariance(self):
        return self._covariance

    @property
    def dimension(self):
        return self._dimension

    def sample(self, n_samps):
        return np.random.multivariate_normal(self._mean, self._covariance, int(n_samps))

    def rotate(self, rotation_matrix):
        """
        Express the MVN in a new frame that is rotated by the rotation matrix
        Args:
            rotation_matrix (rotation matrix):
        """
        self._mean = rotation_matrix @ self._mean
        self._covariance = rotation_matrix @ self._covariance @ rotation_matrix.T
        return

    def change_frame(self, offset_vec, rotation_matrix):
        """
        Change from frame A to frame B.
        Args:
            offset_vec (nx1 numpy array): vector from origin of frame A to frame B
            rotation_matrix (n x n numpy array): rotation matrix corresponding to the 
                                                 angle of the x axis of frame A to frame B
        """
        # By convention, we need to translate before rotating.
        self._mean = self._mean - offset_vec
        self.rotate(rotation_matrix)
        return
        
    def compute_moment(self, multi_idx):
        fun = MultivariateNormal.moment_functions[multi_idx]
        return float(fun(self._t, self._mean, self._covariance))
    
    def compute_moments_up_to(self, max_order):
        """ Compute moments of order up to "max_order".

        Args:
            max_order (int): maximum order of moments to compute.
        """
        # Check if we need to compile moment functions.
        if self._dimension not in MultivariateNormal.compilations_complete.keys():
            MultivariateNormal.compile_moment_functions_up_to(self._dimension, max_order)
        elif MultivariateNormal.compilations_complete[self._dimension] < max_order:
            MultivariateNormal.compile_moment_functions_up_to(self._dimension, max_order)

        # Populate moment_values with moments up to "max_order".
        moment_values = dict()
        for order in range(1, max_order+1):
            # All possible moment multi-indices of the current order.
            possible_multi_idxs = constant_sum_tuples(self._dimension, order)
            for multi_idx in possible_multi_idxs:
                moment_values[multi_idx] = MultivariateNormal.moment_functions[multi_idx](self._t, self._mean, self._covariance)
        MultivariateNormal.compilations_complete[self._dimension] = max_order
        return moment_values
    
    def copy(self):
        return MultivariateNormal(self._mean, self._covariance)

    @classmethod
    def compile_moment_functions_up_to(cls, dimension, max_order):
        """ Compile all moment functions up to a certain order.

        Args:
            max_order (int): maximum order to compile functions up to.
        """
        new_funcs = generate_mvg_moment_functions(max_order, dimension)
        MultivariateNormal.moment_functions.update(new_funcs)
        return