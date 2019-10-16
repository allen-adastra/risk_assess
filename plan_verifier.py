from geom_utils import Ellipse
from models import UncontrolledCarIncremental
import numpy as np
import math
import time

class PlanVerifier(object):
    def __init__(self, x0, y0, v0, theta0, accels, steers, car_coord_ellipse):
        self.deterministic_model = UncontrolledCarIncremental(x0, y0, v0, theta0)

        # Simulate to get self.xs, self.ys, self.thetas
        self.xs, self.ys, self.vs, self.thetas = self.deterministic_model.simulate(x0, y0, v0, theta0, steers, accels)
        self.ellipses = self.generate_ellipses(car_coord_ellipse)

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
            moments: instance of UncontrolledKinematicCarState
        In this function, we think of the random variable z = a1 * x + a2 * y + b
        """
        l = half_space.line
        E_z = l.a1 * moments.E_x + l.a2 * moments.E_y + l.b
        E2_z = l.a1**2 * moments.E2_x + 2 * l.a1 * l.a2 * moments.E_xy + 2 * l.a1 * l.b * moments.E_x + l.a2**2 * moments.E2_y\
              + 2 * l.a2 * l.b * moments.E_y + l.b**2
        return (E2_z - E_z**2)/E2_z
        

    def check_uncertain_agent(self, uncertain_agent, n_lines):
        """
        For the time horizon, find the chebyshev bound!
        """
        ts = np.linspace(0, 2 * math.pi, n_lines)
        half_space_sets = len(self.ellipses) * [None]
        for i, e in enumerate(self.ellipses):
            half_space_sets[i] = set(e.generate_halfspaces_containing_ellipse(ts))
        moments = uncertain_agent.propgate_moments()
        prob_bounds = len(moments) * [None]
        for i in range(len(moments)):
            prob_bounds[i] = min([self.chebyshev_bound_halfspace(hs, moments[i]) for hs in half_space_sets[i]])
        return prob_bounds