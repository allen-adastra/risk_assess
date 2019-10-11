import sympy as sp
from itertools import accumulate
import numpy as np
import math
from random_variables import *
from copy import copy

class InputVariables(object):
  def __init__(self, **attrs):
    for name, value in attrs.items():
      setattr(self, name, value)

class UncontrolledKinematicCarState(object):
    def __init__(self, x0, y0, v0, theta0):
        self.E_x = x0
        self.E_y = y0
        self.E_xy = x0 * y0
        self.E2_x = x0**2
        self.E2_y = y0**2
        self.E_xvs = x0 * v0 * math.sin(theta0)
        self.E_xvc = x0 * v0 * math.cos(theta0)
        self.E_yvs = y0 * v0 * math.sin(theta0)
        self.E_yvc = y0 * v0 * math.cos(theta0)
        self.E_xs = x0 * math.sin(theta0)
        self.E_xc = x0 * math.cos(theta0)
        self.E_ys = y0 * math.sin(theta0)
        self.E_yc = y0 * math.cos(theta0)
        self.E_v = v0
        self.E2_v = v0**2
        self.theta = SumOfRVs(theta0, [])

class UncontrolledCarIncremental(object):
    def __init__(self, x0, y0, v0, theta0):
        self.state = UncontrolledKinematicCarState(x0, y0, v0, theta0)

    def propagate_moments(self, w_theta, w_v):
        """
        State:
            E_x = E[x_t]
            E_y = E[y_t]
            E_xy = E[x_ty_t]
            E2_x = E[x_t^2]
            E2_y = E[y_t^2]
            E_xvs: E[x_t v_t sin(theta_t)]
            E_xvc: E[x_t v_t cos(theta_t)]
            E_yvs: E[y_t v_t sin(theta_t)]
            E_yvc: E[y_t v_t cos(theta_t)]
            E_xs: E[x_t sin(theta_t)]
            E_xc: E[x_t cos(theta_t)]
            E_ys: E[y_t sin(theta_t)]
            E_yc: E[y_t cos(theta_t)]
            E_v: E[v_t]
            E2_v: E[v_t^2]
            theta: instance of "SumOfRVs"
        """
        state = self.state # state is now an alias for self.state
        cos_w_theta = CosSumOfRVs(0, [w_theta])
        sin_w_theta = SinSumOfRVs(0, [w_theta])
        cos_theta = state.theta.cos_applied()
        sin_theta = state.theta.sin_applied()
        E_cos_theta_sin_theta = CrossSumOfRVs(cos_theta.c, cos_theta.random_variables).compute_moment(1)

        # Compute E[x_{t+1} v_{t+1} sin(theta_{t+1})]
        E_xvs_new = state.E_xvs * cos_w_theta.compute_moment(1) + state.E_xvc * sin_w_theta.compute_moment(1)\
                    + state.E2_v * E_cos_theta_sin_theta * cos_w_theta.compute_moment(1)\
                    + state.E2_v * sin_w_theta.compute_moment(1) * cos_w_theta.compute_moment(2)\
                    + state.E_v * w_v.compute_moment(1) * E_cos_theta_sin_theta * cos_w_theta.compute_moment(1)\
                    + state.E_v * w_v.compute_moment(1) * sin_w_theta.compute_moment(1) * cos_theta.compute_moment(2)\
                    + w_v.compute_moment(1) * cos_w_theta.compute_moment(1) * state.E_xs\
                    + w_v.compute_moment(1) * state.E_xc * sin_w_theta.compute_moment(1)

        # Compute E[x_{t+1} v_{t+1} cos(theta_{t+1})]
        E_xvc_new = state.E_xvc * cos_w_theta.compute_moment(1) - state.E_xvc * sin_w_theta.compute_moment(1)\
                    - state.E2_v * E_cos_theta_sin_theta * sin_w_theta.compute_moment(1) + state.E2_v * cos_theta.compute_moment(2) * cos_w_theta.compute_moment(1)\
                    - state.E_v * w_v.compute_moment(1) * E_cos_theta_sin_theta * sin_w_theta.compute_moment(1)\
                    + state.E_v * cos_theta.compute_moment(2) * w_v.compute_moment(1) * cos_w_theta.compute_moment(1)\
                    - w_v.compute_moment(1) * sin_w_theta.compute_moment(1) * state.E_xs + w_v.compute_moment(1) * cos_w_theta.compute_moment(1) * state.E_xc
        
        # Compute E[y_{t+1} v_{t+1} sin(theta_{t+1})]
        E_yvs_new = state.E_yvs * cos_w_theta.compute_moment(1) + state.E_yvc * sin_w_theta.compute_moment(1)\
                    + state.E2_v * sin_theta.compute_moment(2) * cos_w_theta.compute_moment(1)\
                    + state.E2_v * E_cos_theta_sin_theta * sin_w_theta.compute_moment(1)\
                    + state.E_v * w_v.compute_moment(1) * sin_theta.compute_moment(2) * cos_w_theta.compute_moment(1)\
                    + state.E_v * w_v.compute_moment(1) * sin_w_theta.compute_moment(1) * E_cos_theta_sin_theta\
                    + w_v.compute_moment(1) * cos_w_theta.compute_moment(1) * state.E_ys\
                    + w_v.compute_moment(1) * sin_w_theta.compute_moment(1) * state.E_yc

        # Compute E[y_{t+1} v_{t+1} cos(theta_{t+1})]
        E_yvc_new = state.E_yvc * cos_w_theta.compute_moment(1) - state.E_yvs * sin_w_theta.compute_moment(1)\
                    - state.E2_v * sin_theta.compute_moment(2) * sin_w_theta.compute_moment(1)\
                    + state.E2_v * E_cos_theta_sin_theta * cos_w_theta.compute_moment(1)\
                    - state.E_v * w_v.compute_moment(1) * sin_theta.compute_moment(2) * sin_w_theta.compute_moment(1)\
                    + state.E_v * w_v.compute_moment(1) * E_cos_theta_sin_theta * cos_w_theta.compute_moment(1)\
                    - w_v.compute_moment(1) * sin_w_theta.compute_moment(1) * state.E_ys\
                    + w_v.compute_moment(1) * cos_w_theta.compute_moment(1) * state.E_yc

        # Compute E[x_{t+1} sin(theta_{t+1})]
        E_xs_new = state.E_v * cos_w_theta.compute_moment(1) * E_cos_theta_sin_theta + state.E_v * sin_w_theta.compute_moment(1) * cos_theta.compute_moment(2)\
                        + state.E_xs * cos_w_theta.compute_moment(1) + state.E_xc * sin_w_theta.compute_moment(1)

        # Compute E[x_{t+1} cos(theta_{t+1})]
        E_xc_new = -state.E_v * sin_w_theta.compute_moment(1) * E_cos_theta_sin_theta + state.E_v *cos_theta.compute_moment(2) * cos_w_theta.compute_moment(1)\
                    - state.E_xs * sin_w_theta.compute_moment(1) + state.E_xc * cos_w_theta.compute_moment(1)

        # Compute E[y_{t+1} sin(theta_{t+1})]
        E_ys_new = state.E_v * sin_theta.compute_moment(2) * cos_w_theta.compute_moment(1) + state.E_v * sin_w_theta.compute_moment(1) * E_cos_theta_sin_theta\
                    + state.E_ys * cos_w_theta.compute_moment(1) + state.E_yc * sin_w_theta.compute_moment(1)

        # Compute E[y_{t+1} cos(theta_{t+1})]
        E_yc_new = -state.E_v * sin_w_theta.compute_moment(1) * sin_theta.compute_moment(2) + state.E_v * cos_w_theta.compute_moment(1) * E_cos_theta_sin_theta\
                    -state.E_ys * sin_w_theta.compute_moment(1) + state.E_yc * cos_w_theta.compute_moment(1)

        # Compute E[v_{t+1}]
        E_v_new = state.E_v + w_v.compute_moment(1)

        # Compute E[v_{t+1}^2] using the facts that:
        #    Sigma_{x + y} = Sigma_x + Sigma_y
        #    E[(x + y)^2] = Sigma_x + Sigma_y + E[x + y]^2
        E2_v_new = (state.E2_v - state.E_v**2)+ w_v.compute_variance() + E_v_new**2

        E_x_new = state.E_x + state.E_v * cos_theta.compute_moment(1)
        E_y_new = state.E_y + state.E_v * sin_theta.compute_moment(1)

        # Update E[x_{t+1}y_{t+1}]
        E_xy_new = state.E2_v * E_cos_theta_sin_theta + state.E_xvs + state.E_yvc + state.E_xy

        # Update E[x_{t+1}^2]
        E2_x_new = state.E2_v * cos_theta.compute_moment(2) + 2 * state.E_xvc + state.E2_x

        # Update E[y_{t+1}^2]
        E2_y_new = state.E2_v * sin_theta.compute_moment(2) + 2 * state.E_yvs + state.E2_y
        
        state.E_x = E_x_new
        state.E_y = E_y_new
        state.E_xy = E_xy_new
        state.E2_x = E2_x_new
        state.E2_y = E2_y_new
        state.E_xvs = E_xvs_new
        state.E_xvc = E_xvc_new
        state.E_yvs = E_yvs_new
        state.E_yvc = E_yvc_new
        state.E_xs = E_xs_new
        state.E_xc = E_xc_new
        state.E_ys = E_ys_new
        state.E_yc = E_yc_new
        state.E_v = E_v_new
        state.E2_v = E2_v_new
        state.theta.add_rv(w_theta)


class UncontrolledKinematicCar(object):
    def __init__(self, n_steps, dt):
        x0 = sp.symbols('x0', real = True, constant = True)
        y0 = sp.symbols('y0', real = True, constant = True)
        v0 = sp.symbols('v0', real = True, constant = True)
        wvs = sp.symbols('wv0:' + str(n_steps), real = True)
        cos_thetas = sp.symbols('cos_theta0:' + str(n_steps), real = True)
        sin_thetas = sp.symbols('sin_theta0:' + str(n_steps), real = True)
        xs = [x0] + n_steps * [0]
        ys = [y0] + n_steps * [0]
        vs = [v0] + n_steps * [0]
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
        """
        Take an instance of InputVariables and arrange the data in a list consistent with the order
        that input variables are specified in the method "get_input_vars"
        """
        return [input_vars.x0, input_vars.y0, input_vars.v0]

    def construct_cos_sin_theta_rvs(self, theta0, wthetas):
        """
        Given a theta0 and a list of instances of RandomVariable that are wtheta's [wtheta_0, wtheta_1, wtheta_2, ....]
        Return a sequence of instances of CosSumOfRVs and SinSumOfRVs
        """
        cos_thetas = [Constant(math.cos(theta0))] + [CosSumOfRVs(theta0, wthetas[:i]) for i in range(1, len(wthetas))]
        sin_thetas = [Constant(math.sin(theta0))] + [SinSumOfRVs(theta0, wthetas[:i]) for i in range(1, len(wthetas))]
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