import sympy as sp
from itertools import accumulate
import numpy as np
import math
from random_variables import *

class InputVariables(object):
  def __init__(self, **attrs):
    for name, value in attrs.items():
      setattr(self, name, value)
        
class UncontrolledKinematicCar(object):
    def __init__(self, n_steps, dt):
        x0 = sp.symbols('x0', real = True, constant = True)
        y0 = sp.symbols('y0', real = True, constant = True)
        v0 = sp.symbols('v0', real = True, constant = True)
        wvs = sp.symbols('wv0:' + str(n_steps), real = True)
        cos_thetas = sp.symbols('cos_theta0:' + str(n_steps), real = True)
        sin_thetas = sp.symbols('sin_theta0:' + str(n_steps), real = True)
        vs = [v0] + n_steps * [0]
        xs = [x0] + n_steps * [0]
        ys = [y0] + n_steps * [0]
        for i in range(1, n_steps + 1):
            vs[i] = vs[i-1] + dt * wvs[i-1]
            xs[i] = xs[i-1] + dt * vs[i-1] * cos_thetas[i-1]
            ys[i] = ys[i-1] + dt * vs[i-1] * sin_thetas[i-1]
        self.xs = [sp.poly(x, wvs + cos_thetas + sin_thetas) for x in xs]
        self.ys = [sp.poly(y, wvs + cos_thetas + sin_thetas) for y in ys]
        self.x0 = x0
        self.y0 = y0
        self.v0 = v0
        self.dt = dt

    def get_final_state(self):
        return [self.xs[-1], self.ys[-1]]

    def get_input_vars(self):
        return [self.x0, self.y0, self.v0]

    def listify_input_vars(self, input_vars):
        return [input_vars.x0, input_vars.y0, input_vars.v0]

    def construct_cos_sin_theta_rvs(self, theta0, wthetas):
        """
        Given a theta0 and a list of instances of RandomVariable that are wtheta's [wtheta_0, wtheta_1, wtheta_2, ....]
        Return a sequence of instances of CosSumOfRVs and SinSumOfRVs
        """
        cos_thetas = [CosSumOfRVs(theta0, wthetas[:i]) for i in range(len(wthetas) + 1)]
        sin_thetas = [SinSumOfRVs(theta0, wthetas[:i]) for i in range(len(wthetas) + 1)]
        return cos_thetas, sin_thetas

"""
Discrete time kinematic car model that stores symbolic expressions for the
cars state at time steps up to n_steps.
"""
class KinematicCarAccelNoise(object):
    def __init__(self, n_steps, dt):
        x0 = sp.symbols('x0', real = True, constant = True)
        y0 = sp.symbols('y0', real = True, constant = True)
        v0 = sp.symbols('v0', real = True, constant = True)
        # uaw is the accel command multiplied by a random variable for accel noise.
        accel_mult_random = sp.symbols('uaw0:' + str(n_steps), real = True)
        # vs is a list of speeds at times 0, 1,..., n_steps
        dvs = list(accumulate([0] + list(accel_mult_random), lambda cum_sum, next_accel: cum_sum + next_accel))
        vs = [v0] + n_steps * [0]
        xs = [x0] + n_steps * [0]
        ys = [y0] + n_steps * [0]
        thetas = sp.symbols('theta0:' + str(n_steps), real = True, constant = True)
        cos_thetas = [sp.cos(t) for t in thetas]
        sin_thetas = [sp.sin(t) for t in thetas]
        for i in range(1, n_steps + 1):
            vs[i] = vs[i-1] + dt * accel_mult_random[i-1]
            xs[i] = xs[i-1] + dt * vs[i-1] * cos_thetas[i-1]
            ys[i] = ys[i-1] + dt * vs[i-1] * sin_thetas[i-1]
        # Get xs and ys in polynomial form.
        self.xs = [sp.poly(x, accel_mult_random) for x in xs]
        self.ys = [sp.poly(y, accel_mult_random) for y in ys]
        self.n_steps = n_steps
        self.dt = dt
        self.x0 = x0
        self.y0 = y0
        self.v0 = v0
        self.thetas = thetas

    def get_final_state(self):
        return [self.xs[-1], self.ys[-1]]

    def get_input_vars(self):
        return list(self.thetas) + [self.x0, self.y0, self.v0]

    def listify_input_vars(self, input_vars):
        return list(input_vars.thetas) + [input_vars.x0, input_vars.y0, input_vars.v0]

    def simulate_step(self, x1, y1, theta1, v1, uaw, dt):
        x2 = x1 + dt * v1 * math.cos(theta1)
        y2 = y1 + dt * v1 * math.sin(theta1)
        v2 = v1 + dt * uaw
        return [x2, y2, v2]