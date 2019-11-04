from icra_formulation_example import Model
from random_variables import Normal
m = Model(1, 1, 1, 1)
n = Normal(0, 1)
for i in range(10):
    m.update(1, 0.01,n, n , n, n)