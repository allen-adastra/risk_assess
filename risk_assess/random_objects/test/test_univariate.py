import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(dir_path + "/../")
import casadi
import time

def standard_normal_cdf(x):
    """ CDF of the standard normal distribution.

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 0.5*(1+casadi.erf(x/(2**0.5)))

def normal_mgf(t, mu, sigma_sq):
    """ Moment generating function of the normal distribution.

    Args:
        t ([type]): [description]
        mu ([type]): [description]
        sigma_sq ([type]): [description]

    Returns:
        [type]: [description]
    """
    return casadi.exp(mu * t + 0.5 * sigma_sq * t**2.0)

def truncated_normal_mgf(t, mu, sigma_sq, a, b):
    """ Moment generating function of the truncated normal distribution.
    source: https://en.wikipedia.org/wiki/Truncated_normal_distribution

    Args:
        t ([type]): [description]
        mu ([type]): [description]
        sigma_sq ([type]): [description]
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
    sigma = sigma_sq**0.5
    numerator = (standard_normal_cdf(b-sigma*t) - standard_normal_cdf(a-sigma*t))
    return normal_mgf(t, mu, sigma_sq) * (numerator/(standard_normal_cdf(b) - standard_normal_cdf(a)))

def build_moment_functions(n, mgf):
    """ Builds a dictionary of functions

    Args:
        n ([type]): [description]
        mgf ([type]): [description]

    Returns:
        [type]: [description]
    """
    moment_functions = dict()
    for i in range(n):
        moment_fun = casadi.gradient(moment_fun, t)
        moment_functions[i+1] = casadi.Function('g' + str(i+1), [t], [moment_fun])
    return moment_functions

t = casadi.MX.sym('t')
mu = casadi.MX.sym('mu')
sigmasq = casadi.MX.sym('sigmasq')
a = casadi.MX.sym('a')
b = casadi.MX.sym('b')
foo = truncated_normal_mgf(t, mu, sigmasq, a, b)

for i in range(8):
    foo = casadi.gradient(foo, t)
    fun = casadi.Function('g' + str(i+1), [t, mu, sigmasq, a, b], [foo])
