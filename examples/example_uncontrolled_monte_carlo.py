import os 
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)) + '/../')
from random_objects import *
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from models import *
import time

def percent_error(approx, exact):
    return 100 * (abs(approx - exact)/exact)

def compute_percent_errors(sampled_moments, propagated_moments):
    """
    Args:
        sampled_moments: instance of UncontrolledKinematicCarPositionMoments determined via sampling
        propagated_moments: instance of UncontrolledKinematicCarPositionMoments determined via propagation
    """
    E_x_error = percent_error(sampled_moments.E_x, propagated_moments.E_x)
    E_y_error = percent_error(sampled_moments.E_y, propagated_moments.E_y)
    E_xy_error = percent_error(sampled_moments.E_xy, propagated_moments.E_xy)
    E2_x_error = percent_error(sampled_moments.E2_x, propagated_moments.E2_x)
    E2_y_error = percent_error(sampled_moments.E2_y, propagated_moments.E2_y)
    print("E[x] percent error: " + str(E_x_error))
    print("E[y] percent error: " + str(E_y_error))
    print("E[xy] percent error: " + str(E_xy_error))
    print("E[x^2] percent error: " + str(E2_x_error))
    print("E[y^2] percent error: " + str(E2_y_error))

def compute_sample_moments(xs, ys):
    """
    Args:
        xs: n_samples * n_timesteps array of x positions
        ys: n_samples * n_timesteps array of y positions
    Computes, for each time step, the moments:
        E[x]
        E[y]
        E[x^2]
        E[y^2]
        E[xy]
    """
    n_t = xs.shape[1]
    moment_objs = n_t * [None]
    xy = np.multiply(xs, ys)
    x2 = np.power(xs, 2)
    y2 = np.power(ys, 2)
    for i in range(n_t):
        moment_objs[i] = UncontrolledKinematicCarPositionMoments(E_x = np.mean(xs[:,i]),
                        E_y = np.mean(ys[:,i]),
                        E_xy = np.mean(xy[:,i]),
                        E2_x = np.mean(x2[:,i]),
                        E2_y = np.mean(y2[:,i]))
    return moment_objs


p = lambda x,y: 5 * x + 4 * y + 1
n_t = 10
dt = 0.05
x0 = -1.1
y0 = 0.0
v0 = 1.0
theta0 = 0.5
wvs = RandomVector([Normal(1.0, 0.05) for i in range(n_t)])

central_component = MixtureComponent(Normal(0.0, 0.001), 0.6)
left_component = MixtureComponent(Normal(0.3, 0.03), 0.25)
right_component = MixtureComponent(Normal(-0.3, 0.03), 0.15)
mm = MixtureModel([central_component, left_component, right_component])
wthetas = RandomVector([mm for i in range(n_t)])
uncontrolled_car_initial_state = UncontrolledCarState(x0, y0, v0, theta0)
ego_car_initial_state = CarState(x0, y0, v0, theta0)
car_model = UncontrolledCar(uncontrolled_car_initial_state)
prop_moments = car_model.propagate_moments(wthetas, wvs)
xs, ys = car_model.monte_carlo(ego_car_initial_state, wthetas, wvs, int(1e5))
sample_moments = compute_sample_moments(xs, ys)

for sm, pm in zip(sample_moments, prop_moments):
    compute_percent_errors(sm, pm.as_position_moments())
