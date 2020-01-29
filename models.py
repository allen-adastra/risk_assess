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


def propagate_one_step(state, w_v, w_theta):
    c = state.theta.cos_applied()
    s = state.theta.sin_applied()
    cs = CrossSumOfRVs(state.theta.c, state.theta.random_variables)
    E_c = c.compute_moment(1)
    E_s = s.compute_moment(1)
    E2_c = c.compute_moment(2)
    E2_s = s.compute_moment(2)
    E_cs = cs.compute_moment(1)

    # Compute moments of control variables.
    E_cw = CosSumOfRVs(0, [w_theta]).compute_moment(1)
    E_sw = SinSumOfRVs(0, [w_theta]).compute_moment(1)
    E_wv = w_v.compute_moment(1)
    E2_wv = w_v.compute_moment(2)
    
    # Renaming old moments.
    E_v = state.E_v
    E2_v = state.E2_v
    E_xs = state.E_xs
    E_ys = state.E_ys
    E_xc = state.E_xc
    E_yc = state.E_yc
    E_xvs = state.E_xvs
    E_xvc = state.E_xvc
    E_yvs = state.E_yvs
    E_yvc = state.E_yvc
    E_x = state.E_x
    E_y = state.E_y
    E_xy = state.E_xy
    E2_x = state.E2_x
    E2_y = state.E2_y

    #
    new_state = deepcopy(state)

    new_state.theta.add_rv(w_theta)

    new_state.E_v = E_v + E_wv

    new_state.E2_v = E2_v + 2 * E_v * E_wv + E2_wv

    new_state.E_xs = E_v*E_cs*E_cs + E_v*E_sw*E2_c + E_xs*E_cw + E_xc * E_sw

    new_state.E_ys = E_v*E2_s*E_cw + E_v*E_sw*E_cs + E_ys*E_cw + E_yc * E_sw

    new_state.E_xc = -E_v*E_cs * E_sw + E_v*E2_c*E_cw - E_xs*E_sw + E_xc*E_cw

    new_state.E_yc = -E_v*E2_s*E_sw + E_v*E_cs*E_cw - E_ys*E_sw + E_yc*E_cw

    new_state.E_xvs = E2_v*E_cs*E_cw + E2_v*E_sw*E2_c+ E_v*E_wv*E_cs*E_cw + E_v*E_wv*E_sw*E2_c + E_xvs*E_cw +\
          E_xvc * E_sw + E_wv * E_xs * E_cw  + E_wv * E_xc * E_sw

    new_state.E_xvc = -E2_v*E_cs*E_sw + E2_v*E2_c*E_cw- E_v*E_wv*E_cs*E_sw+ E_v*E_wv*E2_c*E_cw -\
          E_xvs*E_sw + E_xvc*E_cw- E_wv*E_xs*E_sw + E_wv*E_xc*E_cw

    new_state.E_yvs = E2_v*E2_s*E_cw+ E2_v*E_cs*E_sw + E_v*E_wv*E2_s*E_cw+ E_v*E_wv*E_cs*E_sw +\
          E_yvs*E_cw+ E_yvc*E_sw + E_wv*E_ys*E_cw+ E_wv*E_yc*E_sw

    new_state.E_yvc = -E2_v*E2_s*E_sw + E2_v*E_cs*E_cw- E_v*E_wv*E2_s*E_sw + E_v*E_wv*E_cs*E_cw-\
          E_yvs*E_sw + E_yvc*E_cw- E_wv*E_ys*E_sw + E_wv*E_yc*E_cw
    
    new_state.E_x = E_x + E_v * E_c

    new_state.E_y = E_y + E_v * E_s

    new_state.E_xy = E2_v * E_cs + E_xvs + E_yvc + E_xy

    new_state.E2_x = E2_v*E2_c + 2*E_xvc + E2_x

    new_state.E2_y = E2_v*E2_s + 2*E_yvs + E2_y

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
        new_state = propagate_one_step(states[i], w_vs[i], w_thetas[i])
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
