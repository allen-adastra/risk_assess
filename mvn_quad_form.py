import numpy as np
from scipy.stats import ncx2, chi2
import scipy.linalg as sla
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import math
import sympy as sp
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
    eigvals, P = np.linalg.eig(CQC)
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
        self._A = A
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
    
###################################################
# Stuff for farming moments to send off to MATLAB.
###################################################
    def compute_moments(self, dmax):
        return [self.compute_moment(i) for i in range(dmax + 1)]

    def dumb_compute_moment_way(self, d, n_samples = 1e6):
        samples = np.random.multivariate_normal(self._mu_x.flatten(), self._Sigma_x, int(n_samples))
        samples = samples.T
        quad_form_samps = (samples.T.dot(self._A)*samples.T).sum(axis=1)
        return np.mean(np.power(quad_form_samps, d))

    def clever_compute_moment_way(self, d):
        # TODO: not working becasue poop.
        xy_moments_needed = 2 * d
        normals = self._mvn.decompose_into_normals(override_independence = True)
        x_moments = normals[0].compute_moments(xy_moments_needed)
        y_moments = normals[1].compute_moments(xy_moments_needed)

        # # We want to essentially perform a whitening transformation.
        # # Let L be the cholesky factor of the covariance matrix.
        # # If x is the original vector, then y = Lx is a whitened random vector.
        # # So we have that L^(-1)y = x.
        # # x^T A x then becomes (L^(-1)y)^T A (L^(-1)y)
        # # Which is equal to y^T A_tilde y where A_tilde = L^(-1)^T A L^(-1)
        # chol_sigma = np.linalg.cholesky(self._Sigma_x)
        # inverse_chol_sigma = np.linalg.inv(chol_sigma)
        # A_tilde = inverse_chol_sigma.T @ self._A @ inverse_chol_sigma

        x = sp.Symbol("x")
        y = sp.Symbol("y")
        vec = sp.Matrix([[x], [y]])
        quad_form = (vec.T @ self._A @ vec)[0]
        quad_form_power = sp.poly(quad_form ** d, [x, y])
        moment = 0
        for coeff, exponent in zip(quad_form_power.coeffs(), quad_form_power.monoms()):
            x_order, y_order = exponent
            moment += coeff * (x_moments[x_order] * y_moments[y_order])
        moment = float(moment)
        return moment

    def compute_moment(self, d):
        if d == 0:
            return 1
        elif d == 1:
            # https://en.wikipedia.org/wiki/Quadratic_form_(statistics)
            return np.trace(self._A @ self._Sigma_x) + (self._mu_x.T @ (self._A @ self._mu_x))[0][0]
        elif d == 2:
            # https://en.wikipedia.org/wiki/Quadratic_form_(statistics)
            if not check_symmetric(self._A):
                A_mat = 0.5 * (self._A + self._A.T)
            else:
                A_mat = self._A
            moment = 2 * np.trace(A_mat @ self._Sigma_x @ A_mat @ self._Sigma_x) + 4 * self._mu_x.T @ A_mat @ self._Sigma_x @ A_mat @ self._mu_x
            return moment[0][0] + self.compute_moment(1)**2
        elif d > 2:
            return self.clever_compute_moment_way(d)

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
    
    def compute_moments(self, dmax):
        return [mvnqf.compute_moments(dmax) for w, mvnqf in self._mvn_components]