from random_variables import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from models import *
import time
import cProfile, pstats, io
from pstats import SortKey
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from plan_verifier import PlanVerifier
from geom_utils import Ellipse

class UncertainAgent(object):
    def __init__(self, x0, y0, v0, theta0, w_thetas, w_vs):
        self.w_thetas = w_thetas
        self.w_vs = w_vs
        self.model = UncontrolledCarIncremental(x0, y0, v0, theta0)

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
x0_ego = -1.1
y0_ego = 0.0
v0_ego = 1.0
theta0_ego = 2.5
x0 = 10
y0 = 10
v0 = 1.0
theta0 = 0.0


"""
Setting up an example uncertain model for the uncontrolled agent uncertain model for speed.
"""
w_vs = RandomVector([Normal(0.1, 0.01) for i in range(n_t)])

# Uncertain model for steering.
central_component = (Normal(0.0, 0.001), 0.6)
left_component = (Normal(0.03, 0.003), 0.25)
right_component = (Normal(-0.03, 0.003), 0.15)
mm = MixtureModel([central_component, left_component, right_component])
w_thetas = RandomVector([mm for i in range(n_t)])
agent = UncertainAgent(x0, y0, v0, theta0, w_thetas, w_vs)

"""
Setting up the plan verifier and an example plan.
"""
accels = n_t * [0.05]
steers = n_t * [0.01]
car_coord_ellipse = Ellipse(1, 3, 0, 0, 0)
verifier = PlanVerifier(x0_ego, y0_ego, v0_ego, theta0_ego, accels, steers, car_coord_ellipse)

t_start = time.time()

pr = cProfile.Profile()
pr.enable()
prob_bounds = verifier.check_uncertain_agent(agent, 10)
pr.disable()
sortby = 'cumtime'
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats()