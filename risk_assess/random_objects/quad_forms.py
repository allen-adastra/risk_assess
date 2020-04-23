import numpy as np
from scipy.stats import ncx2, chi2
import scipy.linalg as sla
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import math
stats = importr('stats')
cqf = importr('CompQuadForm')

def compute_lambdas_deltas(mux, Sigmax, Q, diagonalization_error_tolerance = 1e-10):
    """
    Compute the lambdas and deltas (non-centrality parameters) as described in Duchesne and De Micheaux.
    These lambdas and deltas correspond to those found in Liu et al as well.
    """
    C = np.linalg.cholesky(Sigmax)
    C = C.T
    CQC = C @ Q @ C.T
    _, P = np.linalg.eig(CQC)
    P = P.T
    Lambda = P @ CQC @ P.T
    lambdas = np.diag(Lambda)
    muy = P @ np.linalg.inv(C.T) @ mux
    deltas = np.square(muy)
    return lambdas, deltas

def compute_cks(lambdas, deltas):
    """
    Compute the coefficients "ck" as defined in Liu et al.
    """
    cs = dict()
    for k in range(1, 5):
        lambda_power = np.power(lambdas, k)
        cs[k] = (sum(lambda_power) + k * np.dot(lambda_power, deltas))[0]
    return cs

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

def compute_ncx2_params(t, mu_x, Sigma_x, A):
    lambdas, deltas = compute_lambdas_deltas(mu_x, Sigma_x, A)
    cks = compute_cks(lambdas, deltas)

    # s1 and s2 as defined in Liu et al.
    s1 = cks[3]/(cks[2]**1.5)
    s2 = cks[4]/(cks[2]**2.0)
    acoeff, noncentrality, dof = compute_dof_noncentrality(s1, s2)

    # The mean of Q, mu_Q, is c1
    # The standard deviation of Q, sigma_Q, is sqrt(2 * c2)
    mu_Q = cks[1]
    sigma_Q = (2 * cks[2])**0.5
    mu_chi = dof + noncentrality
    sigma_chi = acoeff * (2**0.5)
    tstar = (t - mu_Q)/sigma_Q
    tspecial = tstar * sigma_chi + mu_chi
    return tspecial, dof, noncentrality
    
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

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
        self._A = np.copy(A)
        self._mu_x = mvn.mean
        self._Sigma_x = mvn.covariance
        self._mvn = mvn

    def upper_tail_probability_monte_carlo(self, t, n_samples = 1e5):
        A = self._A
        mu_x = self._mu_x
        Sigma = self._Sigma_x
        samples = np.random.multivariate_normal(mu_x.flatten(), Sigma, int(n_samples))
        samples = samples.T # 2 x n_samples array
        assert samples.shape[0] == 2
        res = (samples.T.dot(A)*samples.T).sum(axis=1)
        n_true = np.argwhere(res > t).size
        return float(n_true)/float(n_samples)

    def upper_tail_probability_noncentral_chisquare(self, t):
        """
        Approximate the upper tail probability using the noncentral chi square approximation.
        Use the scipy implementation of ncx2.
        """
        tspecial, dof, noncentrality = compute_ncx2_params(t, self._mu_x, self._Sigma_x, self._A)
        if noncentrality == 0.0:
            # If non-centrality is zero, we can just use the standard chi2 distribution.
            return 1.0 - chi2.cdf(tspecial, dof, loc=0, scale=1)
        else:
            return 1.0 - ncx2.cdf(tspecial, dof , noncentrality)

    def upper_tail_probability_noncentral_chisquare_R(self, t):
        """
        Approximate the upper tail probability using the noncentral chi square approximation.
        Use the R implementation of ncx2.
        """
        tspecial, dof, noncentrality = compute_ncx2_params(t, self._mu_x, self._Sigma_x, self._A)
        return 1.0 - stats.pchisq(tspecial, dof , noncentrality)[0]
    
    def upper_tail_probability_noncentral_chisquare_cqf(self, t):
        """
        Approximate the upper tail probability using the noncentral chi square approximation.
        This directly uses the implementation in the CompQuadForm package in R.
        """
        lambdas, deltas = compute_lambdas_deltas(self._mu_x, self._Sigma_x, self._A)
        return cqf.liu(t, ro.FloatVector(list(lambdas)), delta = ro.FloatVector(list(deltas)))[0]

    def upper_tail_probability_imhof(self, t, eps_abs, eps_rel):
        """
        Use the method of imhof which has guaranteed bounds on error.
        """
        lambdas, deltas = compute_lambdas_deltas(self._mu_x, self._Sigma_x, self._A)
        out = cqf.imhof(t, ro.FloatVector(list(lambdas)), delta = ro.FloatVector(list(deltas)), epsabs = eps_abs, epsrel = eps_rel)
        out = dict(zip(out.names, list(out)))
        if out['abserr'][0] > 2 * eps_abs:
            print("Warning! The absolute error is %.3E which is more than double the absolute error tolerance" % out['abserr'][0])
        return out['Qq'][0]

    def upper_tail_probability(self, t, method, **kwargs):
        """
        Compute the probability:
            Prob(x'Ax > t)
        """
        if method == "monte_carlo":
            return self.upper_tail_probability_monte_carlo(t, kwargs['n_samples'])
        elif method == "noncentral_chisquare":
            return self.upper_tail_probability_noncentral_chisquare(t)
        elif method == "noncentral_chisquare_R":
            return self.upper_tail_probability_noncentral_chisquare_R(t)
        elif method == "noncentral_chisquare_cqf":
            return self.upper_tail_probability_noncentral_chisquare_cqf(t)
        elif method == "imhof":
            imhof_prob =  self.upper_tail_probability_imhof(t, kwargs['eps_abs'], kwargs['eps_rel'])
            return imhof_prob
        else:
            raise NotImplementedError("Invalid method name.")
    
    def compute_moments(self, t, dmax):
        """
        Compute the moments of Q(x) - t so that the Chebshev and SOS techniques can be used for
        estimating Prob(Q(x) - t <= 0).
        """
        return [self.compute_moment(t, i) for i in range(dmax + 1)]

    def compute_moment(self, t, d):
        """
        Compute the moments of Q(x) - t so that the Chebshev and SOS techniques can be used for
        estimating Prob(Q(x) - t <= 0).
        """
        if d == 0:
            return 1
        elif d == 1:
            # https://en.wikipedia.org/wiki/Quadratic_form_(statistics)
            return np.trace(self._A @ self._Sigma_x) + (self._mu_x.T @ self._A @ self._mu_x)[0][0] - t
        elif d == 2:
            # https://en.wikipedia.org/wiki/Quadratic_form_(statistics)
            # Note: variance is invariant under translation, so we don't worry about the -1 component.
            if not check_symmetric(self._A):
                A_mat = 0.5 * (self._A + self._A.T)
            else:
                A_mat = self._A
            variance = 2 * np.trace(A_mat @ self._Sigma_x @ A_mat @ self._Sigma_x) + 4 * self._mu_x.T @ A_mat @ self._Sigma_x @ A_mat @ self._mu_x
            return variance[0][0] + self.compute_moment(t, 1)**2
        elif d > 2:
            raise Warning("Using monte carlo to compute moments.")
            return self.monte_carlo_moments(t, d)

    def monte_carlo_moments(self, t, d, n_samples = 1e6):
        """
        Estimate the moments of Q(x) - t with Monte Carlo.
        """
        samples = np.random.multivariate_normal(self._mu_x.flatten(), self._Sigma_x, int(n_samples))
        samples = samples.T
        quad_form_samps = (samples.T.dot(self._A)*samples.T).sum(axis=1)
        quad_form_samps_minus_one = quad_form_samps - t * np.ones(quad_form_samps.shape)
        return np.mean(np.power(quad_form_samps_minus_one, d))

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
        self._mvn_components = [(prob, MvnQuadForm(np.copy(A), mvn)) for prob, mvn in zip(gmm.component_probabilities, gmm.component_random_variables)]
    
    def component_upper_tail_probs(self, t, method, overshoot_one_tolerance = 1e-6, **kwargs):
        """
        Compute the component probabilities:
            P(Q_i > t)
        Where Q_i is a component of Q.
        """
        component_probs = [mvnqf.upper_tail_probability(t, method, **kwargs) for _, mvnqf in self._mvn_components]
        return component_probs

    def upper_tail_probability(self, t, method,  overshoot_one_tolerance = 1e-6, **kwargs):
        """
        Approximate the probability:
            P(Q > t)
        """
        upper_tail_prob = 0
        i = 0
        for component_weight, mvnqf in self._mvn_components:                
            mvnqf_prob = mvnqf.upper_tail_probability(t, method, **kwargs)
            upper_tail_prob += component_weight * mvnqf_prob
            i += 1
        # The calculated probability will have an associated numerical error. Check
        # that the numerical error does not exceed the tolerable amount.
        assert upper_tail_prob < 1.0 + overshoot_one_tolerance
        return min(upper_tail_prob, 1.0)
    
    def compute_moment(self, t, d):
        """
        Compute the dth moment of Q(x) - t.
        """
        return sum([w * mvnqf.compute_moment(t, d) for w, mvnqf in self._mvn_components])