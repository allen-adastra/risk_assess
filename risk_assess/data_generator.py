from scipy import random
import numpy as np
from risk_assess.random_objects.random_variables import *
from risk_assess.random_objects.quad_forms import *

def random_psd_matrix(n):
    """
    Generate a random n dimensional PSD matrix.
    """
    random_matrix = random.rand(n, n)
    psd_matrix = np.dot(random_matrix, random_matrix.transpose())
    return psd_matrix

def generate_random_MVNQF(n):
    """
    Generate a random n dimensional MVNQF.
    """
    A = random_psd_matrix(n)
    Sigma = random_psd_matrix(n)
    mu_x = np.random.rand(n, 1)
    mvn = MultivariateNormal(mu_x, Sigma)
    mvnqf = MvnQuadForm(A, mvn)
    return mvnqf