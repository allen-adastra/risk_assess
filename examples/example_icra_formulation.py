import os 
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)) + '/../')
from random_objects import *
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
import time

class Model(object):
    def __init__(self, x0, y0, v0, theta0):
        self.xs = [x0]
        self.ys = [y0]
        self.vs = [v0]
        self.thetas = [theta0]
        self.steers = [0]
        self.rv_omegas = [Constant(0)] # initialize this to a constant

        # Random variable moments
        self.sigma_x_squared = [0]
        self.sigma_y_squared = [0]
        self.sigma_x_y = [0]
        self.sigma_x_cos = [0]
        self.sigma_x_sin = [0]
        self.sigma_y_cos = [0]
        self.sigma_y_sin = [0]


    def update(self, new_v, steer, rv_v, rv_x, rv_y, rv_omega):
        """
        new_v: control input, new velocity to update to
        steer: control input, the change in theta
        rv_v: random variable for speed
        rv_x: random variable for x position
        rv_y: random variable for y position
        rv_omega: random variable for steer
        """

        sum_steers = sum(self.steers)
        cos_theta = CosSumOfRVs(sum_steers, self.rv_omegas)
        sin_theta = SinSumOfRVs(sum_steers, self.rv_omegas)
        cos_theta_sin_theta = CrossSumOfRVs(sum_steers, self.rv_omegas) #cos(theta) * sin(theta)

        # Update position covariance matrix quantities.
        new_sigma_x_squared = self.sigma_x_squared[-1] + rv_x.compute_variance() + 2 * self.sigma_x_cos[-1] * (new_v + rv_v.compute_moment(1)) \
            + (new_v**2 + 2 * new_v * rv_v.compute_moment(1)) * cos_theta.compute_variance() + rv_v.compute_moment(2) * cos_theta.compute_moment(2)\
            - (rv_v.compute_moment(1)**2) * cos_theta.compute_moment(1)**2
        new_sigma_y_squared = self.sigma_y_squared[-1] + rv_y.compute_variance() + 2 * self.sigma_y_sin[-1] * (new_v + rv_v.compute_moment(1))\
            + (new_v**2 + 2 * new_v * rv_v.compute_moment(1)) * sin_theta.compute_variance() + rv_v.compute_moment(2) * sin_theta.compute_moment(2)\
            - (rv_v.compute_moment(1)**2) * sin_theta.compute_moment(1)**2
        new_sigma_x_y = (new_v**2) * cos_theta_sin_theta.compute_covariance() + 2 * new_v * rv_v.compute_moment(1) * cos_theta_sin_theta.compute_covariance()\
            + rv_v.compute_moment(2) * cos_theta_sin_theta.compute_moment(1) - cos_theta.compute_moment(1) * sin_theta.compute_moment(1) * rv_v.compute_moment(1)**2
        self.sigma_x_squared.append(new_sigma_x_squared)
        self.sigma_y_squared.append(new_sigma_y_squared)
        self.sigma_x_y.append(new_sigma_x_y)

        # Update position-trigonometric covariances
        # TODO: double check the update equations between Ashkan's original paper and the ICRA submission...
        c_omega = CosSumOfRVs(self.steers[-1], [self.rv_omegas[-1]])
        s_omega = SinSumOfRVs(self.steers[-1], [self.rv_omegas[-1]])
        expected_c_omega = c_omega.compute_moment(1)
        expected_s_omega = s_omega.compute_moment(1)
        
        square_mat = np.array([[expected_c_omega, -expected_s_omega],
                                [expected_s_omega, expected_c_omega]])
        expected_v_plus_disturbance = new_v + rv_v.compute_moment(1)
        r1k = expected_v_plus_disturbance * (expected_c_omega * cos_theta.compute_variance() - expected_s_omega * cos_theta_sin_theta.compute_covariance())
        r2k = expected_v_plus_disturbance * (expected_s_omega * sin_theta.compute_variance() + expected_c_omega * cos_theta_sin_theta.compute_covariance())
        r3k = expected_v_plus_disturbance * (expected_c_omega * cos_theta_sin_theta.compute_covariance() - expected_s_omega * sin_theta.compute_variance())
        r4k = expected_v_plus_disturbance * (expected_s_omega * sin_theta.compute_variance() + expected_c_omega * cos_theta_sin_theta.compute_covariance())
        old_x_position_trig_covariances = np.array([[self.sigma_x_cos[-1]], [self.sigma_x_sin[-1]]])
        old_y_position_trig_covariances = np.array([[self.sigma_y_cos[-1]], [self.sigma_y_sin[-1]]])
        new_x_position_trig_covariances = np.matmul(square_mat, old_x_position_trig_covariances) + np.array([[r1k],[r2k]])
        new_y_position_trig_covariances = np.matmul(square_mat, old_y_position_trig_covariances) + np.array([[r3k],[r4k]])
        self.sigma_x_cos.append(new_x_position_trig_covariances[0][0])
        self.sigma_x_sin.append(new_x_position_trig_covariances[1][0])
        self.sigma_y_cos.append(new_y_position_trig_covariances[0][0])
        self.sigma_y_sin.append(new_y_position_trig_covariances[1][0])

        self.steers.append(steer)
        self.rv_omegas.append(rv_omega)
if __name__ == "__main__":
    m = Model(1, 1, 1, 1)
    n = Normal(0, 1)
    for i in range(10):
        m.update(1, 0.01,n, n , n, n)