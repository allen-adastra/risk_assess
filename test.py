from random_vector import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from kinematic_car import KinematicCarAccelNoise
import time

p = lambda x,y: x**2 + y**2 - 1
n_t = 5
dt = 0.05
alpha = 500
beta = 550
x0 = -3.0
y0 = 0.0
v0 = 0
uaccel_seq = 2*np.ones((1,n_t)).flatten()
theta_seq = np.zeros((1,n_t)).flatten()
random_vec = RandomVector([cBetaRandomVariable(alpha, beta, accel) for accel in uaccel_seq])


print("MULTINOMIAL RESULTS")
verification_function = StochasticVerificationFunction(p, KinematicCarAccelNoise, n_t, dt)
verification_function.compile_moment_functions_multinomial()
verification_function.set_problem_data(random_vec, theta_seq, x0, y0, v0)
start_t = time.time()
verification_function.compute_prob_bound_multimonial()
end_t = time.time()
print("Time to compute : " + str(end_t - start_t))