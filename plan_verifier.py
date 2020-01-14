from plan_verification.geom_utils import Ellipse, rotation_matrix
from plan_verification.models import simulate_deterministic
import numpy as np
import math
import time
from plan_verification.mvn_quad_form import GmmQuadForm
from copy import copy, deepcopy

class PlanVerifier(object):
    def __init__(self, initial_state, accels, steers, car_coord_ellipse):
        """
        Args:
            initial_state: instance of CarState
            accels: deterministic control sequence for the ego vehicle
            steers: deterministic control sequence for the ego vehicle
            car_coord_ellipse: instance of Ellipse defining an ellipse in the cars coordinates. x_center, y_center, and theta should be zero.
        """
        # Simulate to get self.xs, self.ys, self.thetas
        xs, ys, vs, thetas = simulate_deterministic(initial_state.x, initial_state.y, initial_state.v, initial_state.theta, steers, accels)
        self.xs = xs.tolist()[0]
        self.ys = ys.tolist()[0]
        self.vs = vs.tolist()[0]
        self.thetas = thetas.tolist()[0]
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

    def chebyshev_bound_halfspace(self, half_space, moments):
        """
        Args:
            half_space: instance of HalfSpace
            moments: instance of UncontrolledCarState
        In this function, we think of the random variable z = a1 * x + a2 * y + b
        """
        l = half_space.line
        E_z = l.a1 * moments.E_x + l.a2 * moments.E_y + l.b
        E2_z = l.a1**2 * moments.E2_x + 2 * l.a1 * l.a2 * moments.E_xy + 2 * l.a1 * l.b * moments.E_x + l.a2**2 * moments.E2_y\
              + 2 * l.a2 * l.b * moments.E_y + l.b**2
        return (E2_z - E_z**2)/E2_z
        

    def assess_risk_moments(self, moments, n_lines):
        """
        Given a list of instances of UncontrolledCarState associated with an uncontrolled agent,
        assess the risk of this given plan.
        Args:
            moments: list of instances of UncontrolledCarState
            n_lines: number of lines to approximate the ellipse with
        """
        ts = np.linspace(0, 2 * math.pi, n_lines)
        ellipses = self.generate_ellipses(self.car_coord_ellipse)
        half_space_sets = len(ellipses) * [None]
        for i, e in enumerate(ellipses):
            half_space_sets[i] = set(e.generate_halfspaces_containing_ellipse(ts))
        prob_bounds = len(moments) * [None]
        for i in range(len(moments)):
            prob_bounds[i] = min([self.chebyshev_bound_halfspace(hs, moments[i]) for hs in half_space_sets[i]])
        return prob_bounds

    def prepare_gmm_quad_forms(self, gmm_traj, method):
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
        gmm_quad_forms = self.prepare_gmm_quad_forms(gmm_traj, method)
        t_prep = time.time() - tstart_prep
        tstart_risk_estimate = time.time()
        risk_estimates = [1 - gmm_quad_form.upper_tail_probability(1, method, **kwargs) for gmm_quad_form in gmm_quad_forms]
        t_risk_assess = time.time() - tstart_risk_estimate
        return risk_estimates, t_prep, t_risk_assess