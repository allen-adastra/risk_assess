"""
Compare moment propagation results from:
    1) TreeRing propagation
    2) Monte Carlo
    3) Linearized Dubins propagation
"""
from risk_assess.uncertain_agent.moment_dynamics import propagate_moments
from risk_assess.uncertain_agent.state_objects import AgentMomentState
from risk_assess.random_objects.random_variables import Normal, cBetaRandomVariable
from risk_assess.deterministic import simulate_deterministic

import numpy as np
import math
import matplotlib.pyplot as plt

def linear_system(v0, theta0, dt):
    # State vector: [x, y, theta, v]^T
    # Control vector: [wtheta, wv]
    # Return A, B, C for the system:
    #   xdot = Ax + Bu + C
    A = dt * np.array([[0.0, 0.0, -v0 * math.sin(theta0), math.cos(theta0)],
                  [0.0, 0.0, v0 * math.cos(theta0), math.sin(theta0)],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
    B = dt * np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])
    C = dt * np.array([[v0 * math.sin(theta0) * theta0],
                  [-v0 * math.cos(theta0) * theta0],
                  [0.0],
                  [0.0]])
    return A, B, C

def linear_system_propagate_moments(initial_state, wvs, wthetas, dt):
    """
    Args:
    """
    mu0 = np.array([[initial_state.E_x], [initial_state.E_y], [initial_state.theta.c], [initial_state.E_v]])
    sigma0 = np.zeros((4, 4))
    A, B, C = linear_system(mu0[3][0], mu0[2][0], dt)
    I_plus_A = np.eye(4) + A
    n_steps = 1 + len(wvs)
    mus = n_steps * [None]
    sigmas = n_steps * [None]
    mus[0] = mu0
    sigmas[0] = sigma0
    moments_all = {"x" : [initial_state.E_x],
                "y" : [initial_state.E_y],
                "x2" : [initial_state.E2_x],
                "y2" : [initial_state.E2_y],
                "xy" : [initial_state.E_xy]}
    for i in range(n_steps - 1):
        wtheta = wthetas[i]
        wv = wvs[i]
        mu_w = np.array([[wtheta.compute_moment(1)], [wv.compute_moment(1)]])
        sigma_w = np.diag([wtheta.compute_variance(), wv.compute_variance()])
        mus[i + 1] = I_plus_A @ mus[i] + B @ mu_w + C
        sigmas[i + 1] = I_plus_A @ sigmas[i] @ I_plus_A.T + B @ sigma_w @ B.T
        E_x = mus[i + 1][0][0]
        E_y = mus[i + 1][1][0]
        moments_all["x"].append(E_x)
        moments_all["y"].append(E_y)
        moments_all["x2"].append(sigmas[i+1][0][0] + E_x**2)
        moments_all["y2"].append(sigmas[i+1][1][1] + E_y**2)
        moments_all["xy"].append(sigmas[i+1][0][1] + E_x * E_y)
    return moments_all

def exact_propagate_moments(initial_state, wvs, wthetas):
    prop_moment_states = propagate_moments(initial_state, wvs, wthetas)
    moments_all = {"x" : [initial_state.E_x],
                   "y" : [initial_state.E_y],
                   "x2" : [initial_state.E2_x],
                   "y2" : [initial_state.E2_y],
                   "xy" : [initial_state.E_xy]}
    for i in range(len(prop_moment_states) - 1):
        moments_all["x"].append(prop_moment_states[i + 1].E_x)
        moments_all["y"].append(prop_moment_states[i + 1].E_y)
        moments_all["x2"].append(prop_moment_states[i + 1].E2_x)
        moments_all["y2"].append(prop_moment_states[i + 1].E2_y)
        moments_all["xy"].append(prop_moment_states[i + 1].E_xy)
    return moments_all

def simulate_mc(initial_state, wvs, wthetas, n_samps, dt = 1):
    accel_samps = np.hstack([rv.sample(n_samps) for rv in wvs])
    steer_samps = np.hstack([rv.sample(n_samps) for rv in wthetas])
    # Note: for the initial state, mean is theta.c
    xs, ys, vs, thetas = simulate_deterministic(initial_state.E_x, initial_state.E_y, initial_state.E_v,
                                                initial_state.theta.c, accel_samps, steer_samps, dt)
    moments_all = {"x" : [initial_state.E_x],
                   "y" : [initial_state.E_y],
                   "x2" : [initial_state.E2_x],
                   "y2" : [initial_state.E2_y],
                   "xy" : [initial_state.E_xy]}
    for i in range(1, xs.shape[1]):
        xs_step = xs[:, i]
        ys_step = ys[:, i]
        moments_all["x"].append(np.mean(xs_step))
        moments_all["y"].append(np.mean(ys_step))
        moments_all["x2"].append(np.mean(np.power(xs_step, 2)))
        moments_all["y2"].append(np.mean(np.power(ys_step, 2)))
        moments_all["xy"].append(np.mean(np.multiply(xs_step, ys_step)))
    return moments_all

def run():
    # Parameters
    n_step = 30
    n_samps = int(1e5)
    dt = 1
    initial_state = AgentMomentState.from_deterministic_state(x0 = 0.0, y0 = 0.0, v0 = 5.0, theta0 = 0.0)

    # Construct random variables.
    wvs = n_step * [cBetaRandomVariable(10, 1000, 1)]
    wthetas = n_step * [Normal(0.04, 1e-3)]

    # Perform exact propagation.
    exact_moments = exact_propagate_moments(initial_state, wvs, wthetas)

    # Perform MOnte Carlo propagation.
    mc_moments = simulate_mc(initial_state, wvs, wthetas, n_samps)

    # Perform linearized propagation.
    linear_moments = linear_system_propagate_moments(initial_state, wvs, wthetas, dt)

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(exact_moments["x2"])
    ax1.plot(mc_moments["x2"], '.')
    ax1.plot(linear_moments["x2"])
    ax1.set_title("$E[x^2]$")
    ax1.legend(["Nonlinear Propagation", "Monte Carlo 1e5 Samples", "Linearized Propagation"])

    ax2.plot(exact_moments["y2"])
    ax2.plot(mc_moments["y2"], '.')
    ax2.plot(linear_moments["y2"])
    ax2.set_title("$E[y^2]$")

    ax3.plot(exact_moments["xy"])
    ax3.plot(mc_moments["xy"], '.')
    ax3.plot(linear_moments["xy"])
    ax3.set_title("$E[xy]$")
    ax3.set_xlabel("Time Step")

    plt.tight_layout()
    plt.show()
run()