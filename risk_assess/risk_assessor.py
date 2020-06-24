from risk_assess.utils import rotation_matrix, change_frame
from risk_assess.random_objects.quad_forms import GmmQuadForm, MvnQuadForm
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

    @staticmethod
    def assess_risk_gmms(body_frame_gmm_traj, A, eps_abs, eps_rel):
        """
        Given a list of gaussian mixture models (GMMs) assess the risk of this plan
        Args:
            gmm_traj (instance of GmmTrajectory)
        Returns:
            list of risks associated to the GMMs.
        """
        risk_estimates = [1 - GmmQuadForm.upper_tail_probability_imhof(gmm, A, 1.0, eps_abs, eps_rel) for gmm in body_frame_gmm_traj]
        return risk_estimates

    @staticmethod
    def assess_risk_gmms_conc(body_frame_gmm_traj, A, inequality):
        """
        Args:
        """
        if inequality == ConcentrationInequality.CANTELLI:
            inequality_func = cantelli
        elif inequality == ConcentrationInequality.VP:
            inequality_func = vp
        elif inequality == ConcentrationInequality.GAUSS:
            inequality_func = gauss
        else: 
            raise Exception("Invalid concentration inequality type.")
        
        t = 1.0 # By convention, we deal with x'Ax - 1 <= 0
        risk_bounds = len(body_frame_gmm_traj) * [None]
        for i, gmm in enumerate(body_frame_gmm_traj):
            # For each GMMQF, compute the overall risk bound by taking the weighted sum of risk bounds on the components.
            gmm_qf_rb = 0.0
            for weight, mvn in gmm:
                first_moment = MvnQuadForm.compute_moment(mvn, A, t, 1)
                second_moment = MvnQuadForm.compute_moment(mvn, A, t, 2)
                variance = second_moment - first_moment**2.0
                gmm_qf_rb += weight * inequality_func(first_moment, variance)
            risk_bounds[i] = gmm_qf_rb
        return risk_bounds