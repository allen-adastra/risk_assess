from random_vector import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from kinematic_car import *
import time

p = lambda x,y: x**2 + 0.1 * y**2 - 1

for i in range(2, 15):
    n_t = i
    dt = 0.05
    alpha = 50
    beta = 55
    x0 = -1.1
    y0 = 0.0
    v0 = 0
    uaccel_seq = 2*np.ones((1,n_t)).flatten()
    theta_seq = np.zeros((1,n_t)).flatten()
    input_vars = InputVariables(x0 = x0, y0 = y0, v0 = v0, thetas = theta_seq)

    random_vec = RandomVector([cBetaRandomVariable(alpha, beta, accel) for accel in uaccel_seq])
    print("Number of time steps: " + str(n_t))
    start_compile = time.time()
    verification_function = StochasticVerificationFunction(p, KinematicCarAccelNoise(n_t, dt))
    verification_function.compile_moment_functions_multinomial()
    print("Time to compile: " + str(time.time() - start_compile))
    start_compute = time.time()
    verification_function.set_random_vector(random_vec)
    verification_function.compute_prob_bound_multimonial(input_vars)
    print("Time to compute: " + str(time.time() - start_compute))
    percent_monte_carlo = verification_function.monte_carlo_result(input_vars, 100000, dt)
    print("Percent fail in monte carlo: " + str(percent_monte_carlo))