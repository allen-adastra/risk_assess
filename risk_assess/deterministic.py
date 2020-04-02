import numpy as np
from dataclasses import dataclass

class Trajectory(object):
    def __init__(self, xs, ys, vs, thetas):
        self._x = xs
        self._y = ys
        self._v = vs
        self._theta = thetas
    
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    
    @property
    def v(self):
        return self._v
    
    @property
    def theta(self):
        return self._theta


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