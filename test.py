from random_vector import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from kinematic_car import KinematicCarAccelNoise

p = lambda x,y: -2*x**2 - 4*y**2 + 3*y**4 + 2.0*x**4 - 0.1
n_t = 4
dt = 0.05
alpha = 500
beta = 550
x0 = -2.0
y0 = 0.0
v0 = 0
uaccel_seq = 2*np.ones((1,n_t)).flatten()
theta_seq = np.zeros((1,n_t)).flatten()
random_vec = RandomVector([cBetaRandomVariable(alpha, beta, accel) for accel in uaccel_seq])


print("MULTINOMIAL RESULTS")
verification_function = StochasticVerificationFunction(p, KinematicCarAccelNoise, n_t, dt)
verification_function.compile_moment_functions_multinomial()
verification_function.set_problem_data(random_vec, theta_seq, x0, y0, v0)
verification_function.compute_rv_moments()