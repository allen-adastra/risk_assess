from geom_utils import Ellipse
from models import simulate_deterministic, UncontrolledCar
import numpy as np
import math
import time
from mvn_quad_form import GmmQuadForm

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
        self.xs, self.ys, self.vs, self.thetas = simulate_deterministic(initial_state, steers, accels)
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

    def assess_risk_gmms(self, gmms):
        """
        Given a list of gaussian mixture models (GMMs) assess the risk of this plan
        Args:
            gmms (list of instances of MixtureModel)
        Returns:
            list of risks associated to the GMMs.
        """
        Q = np.zeros((2, 2))
        Q[0][0] = 1.0/(self.car_coord_ellipse.a**2)
        Q[1][1] = 1.0/(self.car_coord_ellipse.b**2)
        risk_estimates = len(gmms) * [None]
        for i in range(len(gmms)):
            gmm_quad_form = GmmQuadForm(Q, gmms[i])
            risk_estimates[i] = gmm_quad_form.upper_tail_probability(1)
        return risk_estimates