from plan_verification.geom_utils import Ellipse, rotation_matrix, change_frame
from plan_verification.models import simulate_deterministic, propagate_moments
import numpy as np
import math
import time
from plan_verification.mvn_quad_form import GmmQuadForm
from copy import copy, deepcopy
import multiprocessing
from functools import partial
def chebyshev_bound_halfspace(half_space, moments):
    """
    Args:
        half_space: instance of HalfSpace
        moments: instance of UncontrolledCarState
    In this function, we think of the random variable z = a1 * x + a2 * y + b
    """
    l = half_space.line
    E_z = l.a1 * moments.E_x + l.a2 * moments.E_y + l.b
    # Chebyshevs Inequality is only valid for the case when the expected value is greater than zero.
    if E_z > 0:
        E2_z = l.a1**2 * moments.E2_x + 2 * l.a1 * l.a2 * moments.E_xy + 2 * l.a1 * l.b * moments.E_x + l.a2**2 * moments.E2_y\
            + 2 * l.a2 * l.b * moments.E_y + l.b**2
        return (E2_z - E_z**2)/E2_z
    else:
        return None

def evaluate_component(initial_agent_state, half_space_sets, input):
    accel_seq, steer_seq, weight = input
    moments = propagate_moments(initial_agent_state, accel_seq, steer_seq)
    component_probs = len(moments)  * [0]
    for i in range(len(moments)):
        # The function chebyshev_bound_halfspace returns None when Chebyshev doesn't hold.
        probs = [chebyshev_bound_halfspace(hs, moments[i]) for hs in half_space_sets[i]]
        valid_probs = [p for p in probs if p is not None]
        component_probs[i] += weight * min(valid_probs)
    return component_probs

class PlanVerifier(object):
    def __init__(self, xs, ys, vs, thetas, car_coord_ellipse):
        """
        Args:
            initial_state: instance of CarState
            accels: deterministic control sequence for the ego vehicle
            steers: deterministic control sequence for the ego vehicle
            car_coord_ellipse: instance of Ellipse defining an ellipse in the cars coordinates. x_center, y_center, and theta should be zero.
        """
        self.xs = xs
        self.ys = ys
        self.vs = vs
        self.thetas = thetas
        self.car_coord_ellipse = car_coord_ellipse

    def generate_ellipses(self, car_coord_ellipse):
        """
        Given an ellipse described in the car coordinates, generate
        ellipses in the global frame
        """
        cce = car_coord_ellipse
        ellipses = len(self.xs) * [None]
        for i in range(len(self.xs)):
            x = self.xs[i]
            y = self.ys[i]
            t = self.thetas[i]
            ellipses[i] = Ellipse(cce.a, cce.b, x + cce.x_center, y + cce.y_center, t + cce.theta)
        return ellipses


    def assess_risk_monte_carlo(self, gmm_control_sequence, x0, y0, v0, theta0, n_samples):
        """
        Args:
            gmm_control_sequence (instance of GmmControlSequence)
            x0 (scalar): initial x position of the agent
            y0 (scalar): initial y position of the agent
        """
        t_start = time.time()

        # Sample control sequences and propagate them into states.
        accel_samps, steer_samps = gmm_control_sequence.sample(n_samples)
        agent_xs, agent_ys, agent_vs, thetas = simulate_deterministic(x0, y0, v0, theta0, accel_samps, steer_samps, 1.0) # gmm_control_sequence applies dt
        # Propagating controls introduces "extra steps". Limit ourselves to this number of steps.
        n_steps = len(self.xs) - 2

        # Ellipse parameters in car coordinates.
        Q = np.zeros((2, 2))
        Q[0][0] = 1.0/(self.car_coord_ellipse.a**2)
        Q[1][1] = 1.0/(self.car_coord_ellipse.b**2)

        # Risks at each time step.
        risks = n_steps * [None]

        # We will have to transform the position samples into the car coordinates at each time step.
        for i in range(n_steps):
            ego_vehicle_position = np.array([[self.xs[i]],
                                             [self.ys[i]]])
            rot_mat = rotation_matrix(self.thetas[i])

            # Each column is a sample [x; y]
            agent_xys = np.vstack((agent_xs[:, i], agent_ys[:, i]))
            change_frame_func = lambda vec : change_frame(vec, ego_vehicle_position, rot_mat)
            np.apply_along_axis(change_frame_func, 0, agent_xys) # Change the frame of each column
            # Evaluate the number of samples for which x'Qx <= 1 (i.e: collides with ellipse)
            res = (agent_xys.T.dot(Q)*agent_xys.T).sum(axis=1)
            n_collision = np.argwhere(res <= 1.0).size
            risks[i] = float(n_collision)/float(n_samples)
        t_total = time.time() - t_start
        return risks, agent_xs, agent_ys, t_total

    def assess_risk_chebyshev_halfspace(self, gmm_control_sequence, initial_agent_state, n_lines):
        accel_seqs = gmm_control_sequence.array_rep["accels"]
        steer_seqs = gmm_control_sequence.array_rep["steers"]
        weights = gmm_control_sequence.array_rep["weights"]
        # Generate half space sets.
        ts = np.linspace(0, 2 * math.pi, n_lines)
        ellipses = self.generate_ellipses(self.car_coord_ellipse)
        half_space_sets = len(ellipses) * [None]
        for i, e in enumerate(ellipses):
            half_space_sets[i] = set(e.generate_halfspaces_containing_ellipse(ts))

        # Time the time it takes to propagate the moments and apply Chebyshev's inequality.
        pool = multiprocessing.Pool()
        t_start = time.time()
        evaluate_component_func = partial(evaluate_component, initial_agent_state, half_space_sets)
        component_probs = pool.map(evaluate_component_func, list(zip(accel_seqs, steer_seqs, weights)))
        prob_bounds = np.zeros((1, len(component_probs[0])))
        for weight, comp in zip(weights, component_probs):
            prob_bounds += weight * np.asarray(comp)
        prob_bounds = prob_bounds.tolist()[0]
        t_total = time.time() - t_start
        pool.close()
        pool.join()
        return prob_bounds, t_total

    def prepare_gmm_quad_forms(self, gmm_traj):
        gmms = gmm_traj.gmms
        Q = np.zeros((2, 2))
        Q[0][0] = 1.0/(self.car_coord_ellipse.a**2)
        Q[1][1] = 1.0/(self.car_coord_ellipse.b**2)
        gmm_quad_forms = len(gmms) * [None]
        for i in range(len(gmms)):
            ego_vehicle_position = np.array([[self.xs[i]],
                                             [self.ys[i]]])
            rot_mat = rotation_matrix(self.thetas[i])
            gmm = deepcopy(gmms[i])
            gmm.change_frame(ego_vehicle_position, rot_mat)
            gmm_quad_forms[i] = GmmQuadForm(Q, gmm)
        return gmm_quad_forms

    def assess_risk_gmms(self, gmm_traj, method, **kwargs):
        """
        Given a list of gaussian mixture models (GMMs) assess the risk of this plan
        Args:
            gmm_traj (instance of GmmTrajectory): gmm trajectory in the global frame
            method (string): method used to assess risk
        Returns:
            list of risks associated to the GMMs.
        """
        tstart_prep = time.time()
        gmm_quad_forms = self.prepare_gmm_quad_forms(gmm_traj)
        t_prep = time.time() - tstart_prep
        tstart_risk_estimate = time.time()
        risk_estimates = [1 - gmm_quad_form.upper_tail_probability(1, method, **kwargs) for gmm_quad_form in gmm_quad_forms]
        t_risk_assess = time.time() - tstart_risk_estimate
        return risk_estimates, t_prep, t_risk_assess