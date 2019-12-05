from random_objects import *
import numpy as np
from data_generator import *

class TestClass:
    def setup(self):
        pass

    def gmm_quadform(self):
        n = 2
        mu_x = np.random.rand(n, 1)
        Sigma = random_psd_matrix(n)
        A = random_psd_matrix(n)
        mvn1 = MultivariateNormal(mu_x, Sigma)
        mvn2 = MultivariateNormal(mu_x, Sigma)
        mc1 = MixtureComponent(mvn1, 0.1)
        mc2 = MixtureComponent(mvn2, 0.9)
        gmm = MixtureModel([mc1, mc2])
        gmmqf = GmmQuadForm(A, gmm)
        print(gmmqf.upper_tail_probability(1.0))
        print("uyay")

if __name__ == "__main__":
    tc = TestClass()
    tc.gmm_quadform()
