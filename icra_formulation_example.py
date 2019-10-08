from random_variables import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from kinematic_car import *
import time

class Model(object):
    def __init__(self, x0, y0, v0, theta0):
        self.xs = [x0]
        self.ys = [y0]
        self.vs = [v0]
        self.thetas = [theta0]
        self.steers = []
        self.rv_omegas = []

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
        cross_moment_cos_sin = cos_theta_sin_theta.compute_moment(1)

        # Update position covariance matrix quantities.
        new_sigma_x_squared = self.sigma_x_squared[-1] + rv_x.compute_variance() + 2 * self.sigma_x_cos[-1] * (new_v + rv_v.compute_moment(1)) \
            + (new_v**2 + 2 * new_v * rv_v.compute_moment(1)) * cos_theta.compute_variance() + rv_v.compute_moment(2) * cos_theta.compute_moment(2)\
            - (rv_v.compute_moment(1)**2) * cos_theta.compute_moment(1)**2
        new_sigma_y_squared = self.sigma_y_squared[-1] + rv_y.compute_variance() + 2 * self.sigma_y_sin[-1] * (new_v + rv_v.compute_moment(1))\
            + (new_v**2 + 2 * new_v * rv_v.compute_moment(1)) * sin_theta.compute_variance() + rv_v.compute_moment(2) * sin_theta.compute_moment(2)\
            - (rv_v.compute_moment(1)**2) * sin_theta.compute_moment(1)**2
        new_sigma_x_y = (new_v**2) * cross_moment_cos_sin.compute_covariance() + 2 * new_v * rv_v.compute_moment(1) * cross_moment_cos_sin.compute_covariance()\
            + rv_v.compute_moment(2) * cross_moment_cos_sin.compute_moment(1) - cos_theta.compute_moment(1) * sin_theta.compute_moment(1) * rv_v.compute_moment(1)**2

        # TODO: update position-trigonometric covariances
        self.steers.append(steer)
        self.rv_omegas.append(rv_omega)
