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


class UncontrolledKinematicCarPositionMoments(object):
    def __init__(self, E_x, E_y, E_xy, E2_x, E2_y):
        self.E_x = E_x
        self.E_y = E_y
        self.E_xy = E_xy
        self.E2_x = E2_x
        self.E2_y = E2_y

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
    
    def as_position_moments(self):
        return UncontrolledKinematicCarPositionMoments(E_x = self.E_x, E_y = self.E_y, E_xy = self.E_xy, E2_x = self.E2_x, E2_y = self.E2_y)

class UncontrolledCarIncremental(object):
    def __init__(self, x0, y0, v0, theta0):
        self._state = UncontrolledKinematicCarState(x0, y0, v0, theta0)

    @property
    def state(self):
        return copy(self._state)

    def simulate(self, x0, y0, v0, theta0, w_thetas_samp, w_vs_samp):
        """
        Given deterministic outcomes for the random variables, simulate x and y
        """
        # Construct lists of theta and v across the full time horizon
        # theta_t is just the sum theta_0 + theta_1 + ... + theta_{t-1}
        thetas = list(accumulate([theta0] + w_thetas_samp))
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        vs = list(accumulate([v0] + w_vs_samp))
        n_step = len(w_vs_samp) + 1
        xs = n_step * [None]
        ys = n_step * [None]
        xs[0] = x0
        ys[0] = y0
        for i in range(1, n_step):
            xs[i] = xs[i-1] + vs[i-1] * cos_thetas[i-1]
            ys[i] = ys[i-1] + vs[i-1] * sin_thetas[i-1]
        return xs, ys

    def monte_carlo(self, x0, y0, v0, theta0, w_thetas, w_vs, n_samps):
        # w_thetas_samps and w_vs_samps are lists of lists of samples
        # across the full horizon.
        w_thetas_samps = [w_thetas.sample() for i in range(n_samps)]
        w_vs_samps = [w_vs.sample() for i in range(n_samps)]
        n_t = w_vs.dimension() + 1 # Number of time steps
        assert w_vs.dimension() == w_thetas.dimension()
        # Each row represents one "sampled trajectory"
        xs = np.zeros((n_samps, n_t))
        ys = np.zeros((n_samps, n_t))
        # TODO: STORE DAS DATA!
        for i in range(n_samps):            
            xs[i], ys[i] = self.simulate(x0, y0, v0, theta0, w_thetas_samps[i], w_vs_samps[i])
        return xs, ys

    def monte_carlo_onestep(self, x0, y0, w_theta, w_v, n_samps):
        """
        Given an initial position and two random variables
        Assuming we start the car at theta = 0 and v = 0
        Check with Monte Carlo...
        """
        w_v_samps = [w_v.sample() for i in range(n_samps)]
        w_theta_samps = [w_theta.sample() for i in range(n_samps)]
        x_samples = [x0 + v_samp * math.cos(theta_samp) for v_samp, theta_samp in zip(w_v_samps, w_theta_samps)]
        y_samples = [y0 + v_samp * math.sin(theta_samp) for v_samp, theta_samp in zip(w_v_samps, w_theta_samps)]
        xy_samples = [x * y for x,y in zip(x_samples, y_samples)]
        E_x = np.mean(x_samples)
        E_y = np.mean(y_samples)
        E2_x = (1.0/len(x_samples)) * np.sum([x**2 for x in x_samples])
        E2_y = (1.0/len(y_samples)) * np.sum([y**2 for y in y_samples])
        E_xy = np.mean(xy_samples)
        ans = {"E[x]": E_x, "E[y]": E_y, "E[x^2]": E2_x, "E[y^2]": E2_y, "E[xy]": E_xy}
        return ans

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
            theta: theta_t, instance of "SumOfRVs"
        """
        state = self._state # state is now an alias for self._state
        cos_w_theta = CosSumOfRVs(0, [w_theta])
        sin_w_theta = SinSumOfRVs(0, [w_theta])
        cos_theta = state.theta.cos_applied()
        sin_theta = state.theta.sin_applied()

        # Pre compute a bunch of relevant moments
        E_cos_theta_sin_theta = CrossSumOfRVs(cos_theta.c, cos_theta.random_variables).compute_moment(1)
        E_cos_w_theta = cos_w_theta.compute_moment(1)
        E_sin_w_theta = sin_w_theta.compute_moment(1)
        E2_cos_theta = cos_theta.compute_moment(2)
        E2_sin_theta = sin_theta.compute_moment(2)
        E_w_v = w_v.compute_moment(1)

        # Compute E[x_{t+1} v_{t+1} sin(theta_{t+1})]
        E_xvs_new = state.E_xvs * E_cos_w_theta + state.E_xvc * E_sin_w_theta\
                    + state.E2_v * E_cos_theta_sin_theta * E_cos_w_theta\
                    + state.E2_v * E_sin_w_theta * cos_w_theta.compute_moment(2)\
                    + state.E_v * E_w_v * E_cos_theta_sin_theta * E_cos_w_theta\
                    + state.E_v * E_w_v * E_sin_w_theta * E2_cos_theta\
                    + E_w_v * state.E_xs * E_cos_w_theta\
                    + E_w_v * state.E_xc * E_sin_w_theta

        # Compute E[x_{t+1} v_{t+1} cos(theta_{t+1})]
        E_xvc_new = state.E_xvc * E_cos_w_theta - state.E_xvs * E_sin_w_theta\
                    - state.E2_v * E_cos_theta_sin_theta * E_sin_w_theta\
                    + state.E2_v * E2_cos_theta * E_cos_w_theta\
                    - state.E_v * E_w_v * E_cos_theta_sin_theta * E_sin_w_theta\
                    + state.E_v * E2_cos_theta * E_w_v * E_cos_w_theta\
                    - E_w_v * E_sin_w_theta * state.E_xs\
                    + E_w_v * E_cos_w_theta * state.E_xc
        
        # Compute E[y_{t+1} v_{t+1} sin(theta_{t+1})]
        E_yvs_new = state.E_yvs * E_cos_w_theta + state.E_yvc * E_sin_w_theta\
                    + state.E2_v * E2_sin_theta * E_cos_w_theta\
                    + state.E2_v * E_cos_theta_sin_theta * E_sin_w_theta\
                    + state.E_v * E_w_v * E2_sin_theta * E_cos_w_theta\
                    + state.E_v * E_w_v * E_sin_w_theta * E_cos_theta_sin_theta\
                    + E_w_v * E_cos_w_theta * state.E_ys\
                    + E_w_v * E_sin_w_theta * state.E_yc

        # Compute E[y_{t+1} v_{t+1} cos(theta_{t+1})]
        E_yvc_new = state.E_yvc * E_cos_w_theta - state.E_yvs * E_sin_w_theta\
                    - state.E2_v * E2_sin_theta * E_sin_w_theta\
                    + state.E2_v * E_cos_theta_sin_theta * E_cos_w_theta\
                    - state.E_v * E_w_v * E2_sin_theta * E_sin_w_theta\
                    + state.E_v * E_w_v * E_cos_theta_sin_theta * E_cos_w_theta\
                    - E_w_v * E_sin_w_theta * state.E_ys\
                    + E_w_v * E_cos_w_theta * state.E_yc

        # Compute E[x_{t+1} sin(theta_{t+1})]
        E_xs_new = state.E_v * E_cos_w_theta * E_cos_theta_sin_theta\
                    + state.E_v * E_sin_w_theta * E2_cos_theta\
                    + state.E_xs * E_cos_w_theta + state.E_xc * E_sin_w_theta

        # Compute E[x_{t+1} cos(theta_{t+1})]
        E_xc_new = -state.E_v * E_sin_w_theta * E_cos_theta_sin_theta\
                + state.E_v *E2_cos_theta * E_cos_w_theta\
                - state.E_xs * E_sin_w_theta + state.E_xc * E_cos_w_theta

        # Compute E[y_{t+1} sin(theta_{t+1})]
        E_ys_new = state.E_v * E2_sin_theta * E_cos_w_theta\
                + state.E_v * E_sin_w_theta * E_cos_theta_sin_theta\
                + state.E_ys * E_cos_w_theta + state.E_yc * E_sin_w_theta

        # Compute E[y_{t+1} cos(theta_{t+1})]
        E_yc_new = -state.E_v * E_sin_w_theta * E2_sin_theta\
                + state.E_v * E_cos_w_theta * E_cos_theta_sin_theta\
                -state.E_ys * E_sin_w_theta + state.E_yc * E_cos_w_theta

        # Compute E[v_{t+1}]
        E_v_new = state.E_v + E_w_v

        # Compute E[v_{t+1}^2] using the facts that:
        #    Sigma_{x + y} = Sigma_x + Sigma_y
        #    E[(x + y)^2] = Sigma_x + Sigma_y + E[x + y]^2
        E2_v_new = (state.E2_v - state.E_v**2) + w_v.compute_variance() + E_v_new**2

        E_x_new = state.E_x + state.E_v * cos_theta.compute_moment(1)
        E_y_new = state.E_y + state.E_v * sin_theta.compute_moment(1)

        # Update E[x_{t+1}y_{t+1}]
        E_xy_new = state.E2_v * E_cos_theta_sin_theta + state.E_xvs + state.E_yvc + state.E_xy

        # Update E[x_{t+1}^2]
        E2_x_new = state.E2_v * E2_cos_theta + 2 * state.E_xvc + state.E2_x

        # Update E[y_{t+1}^2]
        E2_y_new = state.E2_v * E2_sin_theta + 2 * state.E_yvs + state.E2_y
        
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