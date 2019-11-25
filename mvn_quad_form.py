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
    Given s1 and s2 as defined inthe paper, compute a, noncentrality, dof
    """
    if s1**2 > s2:
        a = 1.0/(s1 - (s1**2 - s2)**0.5)
        noncentrality = s1 * a**3 - a**2
        dof = a**2 - 2 * noncentrality
        assert dof > 0
        assert noncentrality > 0
    else:
        a = 1/s1
        noncentrality = 0
        dof = 1/s1**2
        assert dof > 0
    return a, noncentrality, dof

"""
This random variable is defined by the following quadratic form:
    Q = x'Ax
Where x is some multivariate normal and A is a PSD matrix. This is an implementation
of the approximation method found in:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.5737&rep=rep1&type=pdf
Which tries to approximate this quadratic form with a non-central chi squared distribution.
"""
class MvnQuadForm(object):
    def __init__(self, A, mu_x, Sigma_x):
        """
        Args:
            A: PSD matrix as described above.
            mu_x: mean vector of the random variable x
            Sigma_x: covariance matrix of the random variable x
        """
        self._A = A
        self._mu_x = mu_x
        self._Sigma_x = Sigma_x

        cks = compute_ck(A, Sigma_x, mu_x, 4)

        # s1 and s2 as defined in the paper
        s1 = cks[3]/(cks[2]**1.5)
        s2 = cks[4]/(cks[2]**2.0)
        a, self._noncentrality, self._dof = compute_dof_noncentrality(s1, s2)


        # The mean of Q, mu_Q, is c1
        # The standard deviation of Q, sigma_Q, is sqrt(2 * c2)
        self._mu_Q = cks[1]
        self._sigma_Q = (2 * cks[2])**0.5
        self._mu_chi = self._dof + self._noncentrality
        self._sigma_chi = a * (2**0.5)


        # We only need ck up to k = 4 for our approximation.
        self._cks = compute_ck(A, Sigma_x, mu_x, 4)

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