import sympy as sp
from itertools import accumulate
import numpy as np
import math
from plan_verification.random_objects import *
from copy import copy, deepcopy
import time

def simulate_deterministic(x0, y0, v0, theta0, accels, steers, dt):
    """
    Given an initial state and control sequences, forward simulate and return future states.
    Args:
        x0 (scalar)
        y0 (scalar)
        v0 (scalar)
        theta0 (scalar)
        accels (n_samps * n_step numpy array)
        steers (n_samps * n_step numpy array)
        dt (scalar)
    """
    assert (steers.shape == accels.shape)
    if len(steers.shape) == 1 and len(accels.shape) == 1:
        # In this case, steers and accels are 1-D arrays, convert them to 2-D
        steers = steers.reshape(1, steers.shape[0])
        accels = accels.reshape(1, accels.shape[0])
    n_samps = steers.shape[0]
    # Repeat elements to correspond to samples
    x0_rep = np.repeat(x0, n_samps).reshape(n_samps, 1)
    y0_rep = np.repeat(y0, n_samps).reshape(n_samps, 1)
    v0_rep = np.repeat(v0, n_samps).reshape(n_samps, 1)
    theta0_rep = np.repeat(theta0, n_samps).reshape(n_samps, 1)

    # Compute headings and speeds
    thetas = np.cumsum(np.hstack((theta0_rep, dt * steers)), axis = 1) # Sum along the rows to get headings
    vs = np.cumsum(np.hstack((v0_rep, dt * accels)), axis = 1) # Sum along the rows to get speeds
    dt_vs = dt * vs

    # Arrive at dt * v_t * cos(theta_t) and dt * v_t * sin(theta_t) for every time step for every sample
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    dt_vs_cos_thetas = np.multiply(dt_vs, cos_thetas)
    dt_vs_sin_thetas = np.multiply(dt_vs, sin_thetas)

    # x and y positions are just the cumulative sums of the v * cos(theta) and v * sin(theta) over time
    xs = np.cumsum(np.hstack((x0_rep, dt_vs_cos_thetas)), axis = 1)
    ys = np.cumsum(np.hstack((y0_rep, dt_vs_sin_thetas)), axis = 1)

    # xs and ys have one extra element compared to vs and thetas.
    xs = xs[:, :-1]
    ys = ys[:, :-1]
    return xs, ys, vs, thetas

def propagate_one_step(state, w_theta, w_v):
    """
    Propagates UncontrolledCarState one step forward in time.
    Args:
        state (instance of UncontrolledCarState)
        w_theta (instance of RandomVariable)
        w_v (instance of RandomVariable)
    """
    cos_theta = state.theta.cos_applied()
    sin_theta = state.theta.sin_applied()
    cos_w_theta = CosSumOfRVs(0, [w_theta])
    # Pre compute a bunch of relevant moments
    E_cos_theta_sin_theta = CrossSumOfRVs(cos_theta.c, cos_theta.random_variables).compute_moment(1)
    E_cos_w_theta = cos_w_theta.compute_moment(1)
    E2_cos_w_theta = cos_w_theta.compute_moment(2)
    E_sin_w_theta = SinSumOfRVs(0, [w_theta]).compute_moment(1)
    E2_cos_theta = cos_theta.compute_moment(2)
    E2_sin_theta = sin_theta.compute_moment(2)
    E_w_v = w_v.compute_moment(1)
    
    # Compute E[x_{t+1} v_{t+1} sin(theta_{t+1})]
    E_xvs_new = state.E_xvs * E_cos_w_theta + state.E_xvc * E_sin_w_theta\
                + state.E2_v * E_cos_theta_sin_theta * E_cos_w_theta\
                + state.E2_v * E_sin_w_theta * E2_cos_w_theta\
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
    
    new_state = deepcopy(state)
    new_state.E_x = E_x_new
    new_state.E_y = E_y_new
    new_state.E_xy = E_xy_new
    new_state.E2_x = E2_x_new
    new_state.E2_y = E2_y_new
    new_state.E_xvs = E_xvs_new
    new_state.E_xvc = E_xvc_new
    new_state.E_yvs = E_yvs_new
    new_state.E_yvc = E_yvc_new
    new_state.E_xs = E_xs_new
    new_state.E_xc = E_xc_new
    new_state.E_ys = E_ys_new
    new_state.E_yc = E_yc_new
    new_state.E_v = E_v_new
    new_state.E2_v = E2_v_new
    new_state.theta.add_rv(w_theta)
    return new_state
    
def propagate_moments(initial_state, w_vs, w_thetas):
    """
    Given some initial 
    Args:
        state (instance of UncontrolledCarState)
        w_vs (list of instances of RandomVariable)
        w_thetas (list of instances of RandomVariable)
    """
    states = [initial_state]
    assert len(w_thetas) == len(w_vs)
    for i in range(len(w_thetas)):
        new_state = propagate_one_step(states[i], w_thetas[i], w_vs[i])
        states.append(new_state)
    return states

class InputVariables(object):
  def __init__(self, **attrs):
    for name, value in attrs.items():
      setattr(self, name, value)

class CarState(object):
    def __init__(self, x0, y0, v0, theta0):
        self.x = x0
        self.y = y0
        self.v = v0
        self.theta = theta0

class UncontrolledCarState(object):
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
    
    def speed_scaled(self, speed_scale_factor):
        return UncontrolledCarState(self.E_x, self.E_y, self.E_v * speed_scale_factor, self.theta.c)
