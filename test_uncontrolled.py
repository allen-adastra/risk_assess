from random_variables import RandomVector, cBetaRandomVariable
import numpy as np
from stochastic_verification_functions import StochasticVerificationFunction
from models import *
import time
import cProfile, pstats, io
from pstats import SortKey


p = lambda x,y: 5 * x + 4 * y + 1
n_t = 20
dt = 0.05
x0 = -1.1
y0 = 0.0
v0 = 0
theta0 = 0.0
wvs = [Normal(0.1, 0.02) for i in range(n_t)]
central_component = (Normal(0.0, 0.1), 0.9)
left_component = (Normal(0.3, 0.05), 0.08)
right_component = (Normal(-0.2, 0.05), 0.02)
mm = MixtureModel([central_component, left_component, right_component])
wthetas = [mm for i in range(n_t)]
car_model = UncontrolledCarIncremental(x0, y0, v0, theta0)
t_start = time.time()
for i in range(len(wvs)):
    car_model.propagate_moments(wthetas[i], wvs[i])
print("Total time: " + str(time.time() - t_start))