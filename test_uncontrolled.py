from random_vector import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from kinematic_car import *
import time

"""
Test the uncontrolled car model.
"""

p = lambda x,y: x**2 + 0.1 * y**2 - 1

for i in range(2, 15):
    n_t = i
    dt = 0.05
    alpha = 50
    beta = 55
    x0 = -1.1
    y0 = 0.0
    v0 = 0
    car_model = UncontrolledKinematicCar(i, dt)
    print("Number of time steps: " + str(n_t))
    start_compile = time.time()
    verification_function = StochasticVerificationFunction(p, car_model)
    verification_function.compile_moment_functions_multinomial()
    print("Time to compile: " + str(time.time() - start_compile))
    print(verification_function.p)