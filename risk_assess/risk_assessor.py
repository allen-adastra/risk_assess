from risk_assess.geom_utils import Ellipse, rotation_matrix, change_frame
from risk_assess.deterministic import simulate_deterministic
from risk_assess.uncertain_agent.moment_dynamics import propagate_moments
from risk_assess.random_objects.quad_forms import GmmQuadForm
from risk_assess.concentration_inequalities import *
import numpy as np
import math
import time
from copy import copy, deepcopy
import multiprocessing
from functools import partial
import scipy.io

def chebyshev_bound_halfspace(half_space, moments):
    """
    Args:
        half_space: instance of HalfSpace
        moments: instance of AgentMomentState
    In this function, we think of the random variable z = a1 * x + a2 * y + b
    """
    l = half_space.line
    E_z = l.a1 * moments.E_x + l.a2 * moments.E_y + l.b
    #Chebyshevs Inequality is only valid for the case when the expected value is greater than zero.
    if E_z > 0:
        variance_x = moments.E2_x - moments.E_x**2
        variance_y = moments.E2_y - moments.E_y**2
        assert variance_x >= 0.0
        assert variance_y >= 0.0
        covariance_xy = moments.E_xy - moments.E_x * moments.E_y
        variance_z = (l.a1**2) * variance_x + (l.a2**2) * variance_y + 2 * l.a1 * l.a2 * covariance_xy
        E2_z = variance_z + E_z**2
        return variance_z/E2_z
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
        if valid_probs:
            component_probs[i] = weight * min(valid_probs)
        else:
            component_probs[i] = None
    return component_probs

class RiskAssessor(object):
    def __init__(self, xs, ys, vs, thetas):
        """
        Args:
            xs : sequence of x positions
            ys : seqeunce of y positions
            vs : sequence of speeds
            thetas : sequence of headings in the global frame
        """
        self.xs = xs
        self.ys = ys
        self.vs = vs
        self.thetas = thetas

    def generate_ellipses(self, car_coord_ellipse):
        """
        Given an ellipse described in the car coordinates, generate
        ellipses parameterized in the global frame.
        """
        cce = car_coord_ellipse
        ellipses = len(self.xs) * [None]
        for i in range(len(self.xs)):
            x = self.xs[i]
            y = self.ys[i]
            t = self.thetas[i]
            ellipses[i] = Ellipse(cce.a, cce.b, x + cce.x_center, y + cce.y_center, t + cce.theta)
        return ellipses

    def assess_risk_gmms(self, gmm_quad_forms, method, **kwargs):
        """
        Given a list of gaussian mixture models (GMMs) assess the risk of this plan
        Args:
            gmm_quad_forms (list of instances of GmmQuadForm):
            method (string): method used to assess risk
        Returns:
            list of risks associated to the GMMs.
        """
        risk_estimates = [1 - gmm_quad_form.upper_tail_probability(1, method, **kwargs) for gmm_quad_form in gmm_quad_forms]
        return risk_estimates

    def assess_risk_gmms_conc(self, gmm_quad_forms, inequality):
        """
        Args:
            gmm_traj ([type]): [description]
            inequality (ConcentrationInequality): [description]
        """
        if inequality == ConcentrationInequality.CANTELLI:
            inequality_func = cantelli
        elif inequality == ConcentrationInequality.VP:
            inequality_func = vp
        elif inequality == ConcentrationInequality.GAUSS:
            inequality_func = gauss
        else: 
            raise Exception("Invalid concentration inequality type.")

        risk_bounds = len(gmm_quad_forms) * [None]
        for i, gmm_qf in enumerate(gmm_quad_forms):
            # For each GMMQF, compute the overall risk bound
            # by taking the weighted sum of risk bounds on the components.
            gmm_qf_rb = 0.0
            for weight, mvnqf in gmm_qf.mvn_components:
                first_moment = mvnqf.compute_moment(1, 1)
                second_moment = mvnqf.compute_moment(1, 2)
                variance = second_moment - first_moment**2.0
                gmm_qf_rb += weight * inequality_func(first_moment, variance)
            risk_bounds[i] = gmm_qf_rb
        return risk_bounds

    def gmm_quad_form_moments_to_matfile(self, gmm_quad_forms, directory, scenario_number):
        n_components = len(gmm_quad_forms[0]._mvn_components)
        traj_components = n_components * [{"weight" : None, "moments" : []}]
        weights = [w for w,_ in gmm_quad_forms[0]._mvn_components]
        for w, component in zip(weights, traj_components):
            component["weight"] = w
        all_gmm_qf_moments = [gmm_qf.compute_moments(5) for gmm_qf in gmm_quad_forms]
        for i in range(n_components):
            traj_components[i]["moments"] = [gmm_qf_moments[i] for gmm_qf_moments in all_gmm_qf_moments]
        for i, comp in enumerate(traj_components):
            filename = directory + "/" + "position_gmm_component_scenario_" + str(scenario_number) + "_component_" + str(i) + ".mat"
            scipy.io.savemat(filename, comp)