from random_variables import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from kinematic_car import *
import time
import cProfile, pstats, io
from pstats import SortKey
"""
Test the uncontrolled car model.
"""

p = lambda x,y: 5 * x + 4 * y + 1
n_t = 10
dt = 0.05
x0 = -1.1
y0 = 0.0
v0 = 0
theta0 = 0.0
wvs = [Normal(1.0, 1.0) for i in range(n_t)]
wthetas = [Normal(1.0, 1.0) for i in range(n_t)]
car_model = UncontrolledCarIncremental(x0, y0, v0, theta0)
for i in range(n_t):
    car_model.propagate_moments(wthetas[i], wvs[i])