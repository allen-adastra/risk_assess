import numpy as np
from scipy.stats import ncx2, chi2

def compute_ck(A, Sigma, mu_x, k_max):
    """
    Compute the coefficients "ck" as defined
    Args:
        A:
        Sigma:
        mu_x:
    """
    # ASigma is A * Sigma
    ASigma = np.matmul(A, Sigma)
    ck = dict()
    # Cache the matrix powers of ASigma in this dictionary
    # Where the key k corresponds to (A * Sigma)^k
    ASigma_powers = dict()
    ASigma_powers[0] = np.eye(ASigma.shape[0]) # The zeroth power of a matrix is the identity.
    for k in range(1, k_max + 1):
        ASigma_powers[k] = np.matmul(ASigma, ASigma_powers[k-1])
        Amu_x = np.matmul(A, mu_x)
        temp = np.matmul(ASigma_powers[k-1], Amu_x)
        ck[k] = np.trace(ASigma_powers[k]) + k * (np.matmul(mu_x.T, temp))[0][0]
    return ck

def compute_dof_noncentrality(s1, s2):
    """
    Given s1 and s2 as defined in the paper, compute acoeff, noncentrality, dof
    """
    if s1**2 > s2:
        acoeff = 1.0/(s1 - (s1**2 - s2)**0.5)
        noncentrality = s1 * acoeff**3 - acoeff**2
        dof = acoeff**2 - 2 * noncentrality
        assert dof > 0
        assert noncentrality > 0
    else:
        acoeff = 1/s1
        noncentrality = 0
        dof = 1/s1**2
        assert dof > 0
    return acoeff, noncentrality, dof

"""
This random variable is defined by the following quadratic form:
    Q = x'Ax
Where x is some multivariate normal and A is a positive definite matrix. This is an implementation
of the approximation method found in:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.5737&rep=rep1&type=pdf
Which tries to approximate this quadratic form with a non-central chi squared distribution.
"""
class MvnQuadForm(object):
    def __init__(self, A, mvn):
        """
        Args:
            A: positive definite matrix as described above.
            mvn: instance of MultivariateNormal
        """
        self._A = A
        self._mu_x = mvn.mean
        self._Sigma_x = mvn.covariance

        cks = compute_ck(A, mvn.covariance, mvn.mean, 4)

        # s1 and s2 as defined in the paper
        s1 = cks[3]/(cks[2]**1.5)
        s2 = cks[4]/(cks[2]**2.0)
        acoeff, self._noncentrality, self._dof = compute_dof_noncentrality(s1, s2)

        # The mean of Q, mu_Q, is c1
        # The standard deviation of Q, sigma_Q, is sqrt(2 * c2)
        self._mu_Q = cks[1]
        self._sigma_Q = (2 * cks[2])**0.5
        self._mu_chi = self._dof + self._noncentrality
        self._sigma_chi = acoeff * (2**0.5)

        # We only need ck up to k = 4 for our approximation.
        self._cks = compute_ck(A, mvn.covariance, mvn.mean, 4)

    def upper_tail_probability_monte_carlo(self, t, n_samples = 1e5):
        A = self._A
        mu_x = self._mu_x
        Sigma = self._Sigma_x
        samples = np.random.multivariate_normal(mu_x.flatten(), Sigma, int(n_samples))
        samples = samples.T
        res = (samples.T.dot(A)*samples.T).sum(axis=1)
        n_true = np.argwhere(res > t).size
        return float(n_true)/float(n_samples)

    def upper_tail_probability(self, t):
        """
        Approximate the probability:
            Prob(x'Ax > t)
        Args:
            A (numpy array): as defined
            mu_x (numpy array): mean vector of x
            Sigma (numpy array): covariance matrix of x
            t (scalar): as defined
        """
        tstar = (t - self._mu_Q)/self._sigma_Q
        tspecial = tstar * self._sigma_chi + self._mu_chi
        if self._noncentrality == 0.0:
            # If non-centrality is zero, we can just use the standard chi2 distribution.
            return 1.0 - chi2.cdf(tspecial, self._dof, loc=0, scale=1)
        else:
            return 1.0 - ncx2.cdf(tspecial, self._dof , self._noncentrality)

class GmmQuadForm(object):
    """
    This random variable is defined by:
        Q = x'Ax
    Where x is some Gaussian Mixture Model (GMM) and A is a positive definite matrix.
    This is an extension of MvnQuadForm.
    """
    def __init__(self, A, gmm):
        """
        Args:
            A (numpy array): as defined
            gmm (instance of MixtureModel with instances of MultivariateNormal as components)
        """
        self._mvn_components = [(prob, MvnQuadForm(A, mvn)) for prob, mvn in zip(gmm.component_probabilities, gmm.component_random_variables)]
    
    def upper_tail_probability(self, t, overshoot_one_tolerance = 1e-6):
        """
        Approximate the probability:
            P(Q > t)
        """
        upper_tail_prob = 0
        for component_prob, mvnqf in self._mvn_components:
            upper_tail_prob += component_prob * mvnqf.upper_tail_probability(t)
        # The calculated probability will have an associated numerical error. Check
        # that the numerical error does not exceed the tolerable amount.
        assert upper_tail_prob < 1.0 + overshoot_one_tolerance
        return min(upper_tail_prob, 1.0)