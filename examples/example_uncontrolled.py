import os 
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)) + '/../')
from random_objects import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from models import *
import time
from plan_verifier import PlanVerifier
from geom_utils import Ellipse

class UncertainAgent(object):
    def __init__(self, initial_state, w_thetas, w_vs):
        """
        Args:
            initial_state: instance of 
        """
        self.w_thetas = w_thetas
        self.w_vs = w_vs
        self.model = UncontrolledCar(initial_state)

    def propgate_moments(self):
        moments = self.model.propagate_moments(self.w_thetas, self.w_vs)
        return moments

def fire_up_matmul():
    # for whatever reason, matmul is really slow the first time we call it
    theta = 0.5
    rotation_matrix = np.array([[math.cos(theta), math.sin(theta)],
                [-math.sin(theta), math.cos(theta)]])
    thing = np.ones((2, 10))
    np.matmul(rotation_matrix, thing)

fire_up_matmul()
# Specify problem parameters.
n_t = 20
dt = 0.05
ego_initial_state = CarState(x0 = -1.1, y0 = 0.0, v0 = 1.0, theta0 = 2.5)
uncontrolled_initial_state = UncontrolledCarState(10, 10, 1.0, 0.0)


"""
Setting up an example uncertain model for the uncontrolled agent uncertain model for speed.
"""
w_vs = RandomVector([Normal(0.1, 0.01) for i in range(n_t)])
# Uncertain model for steering.
central_component = MixtureComponent(Normal(0.0, 0.001), 0.6)
left_component = MixtureComponent(Normal(0.03, 0.003), 0.25)
right_component = MixtureComponent(Normal(-0.03, 0.003), 0.15)
mm = MixtureModel([central_component, left_component, right_component])
w_thetas = RandomVector([mm for i in range(n_t)])
agent = UncertainAgent(uncontrolled_initial_state, w_thetas, w_vs)

"""
Setting up the plan verifier and an example plan.
"""
accels = n_t * [0.05]
steers = n_t * [0.01]
car_coord_ellipse = Ellipse(1, 3, 0, 0, 0)
verifier = PlanVerifier(ego_initial_state, accels, steers, car_coord_ellipse)
prob_bounds = verifier.check_uncertain_agent(agent, 10)