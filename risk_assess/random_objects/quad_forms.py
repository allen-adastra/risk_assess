import numpy as np
from scipy.stats import ncx2, chi2
import scipy.linalg as sla
import math
import glob
import ctypes
import os
import warnings
dir_path = os.path.dirname(os.path.realpath(__file__))

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
        cs[k] = (sum(lambda_power) + k * np.dot(lambda_power, deltas))
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
Where x is some multivariate normal and A is a positive definite matrix.
The methods implemented here can be found in the following and its referencecs:
    Duchesne, Pierre, and Pierre Lafaye De Micheaux. "Computing the distribution of quadratic forms: 
    Further comparisons between the Liu–Tang–Zhang approximation and exact methods." Computational
    Statistics & Data Analysis 54.4 (2010): 858-862.
"""
class MvnQuadForm(object):
    #
    # Load the C++ library.
    #
    # find the shared library, the path depends on the platform and Python version
    libfile = glob.glob(dir_path + '/../../build/*/imhof*.so')[0]

    # Open the shared library
    imhof = ctypes.CDLL(libfile)

    # Alias to shorten name.
    double_ptr = np.ctypeslib.ndpointer(dtype=np.float64)

    # Signature
    # upper_tail_prob(const double &x, double *lambda, const int &lambdalen, double *h, double *delta2, double *Qx,
    #                  double *epsabs_out, const double &epsabs, const double &epsrel, const int &limit)
    # Tell ctypes the argument and result types.
    imhof.upper_tail_prob.argtypes = [ctypes.c_double, double_ptr, ctypes.c_int, double_ptr, double_ptr, double_ptr\
                                ,double_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    imhof.upper_tail_prob.restypes = [ctypes.c_void_p]
    
    @staticmethod
    def upper_tail_probability_monte_carlo(mvn, A, t, **kwargs):
        n_samples = kwargs["n_samples"]
        samples = np.random.multivariate_normal(mvn.mean, mvn.covariance, int(n_samples))
        samples = samples.T # 2 x n_samples array
        assert samples.shape[0] == 2
        res = (samples.T.dot(A)*samples.T).sum(axis=1)
        n_true = np.argwhere(res > t).size
        return float(n_true)/float(n_samples)
    
    @staticmethod
    def upper_tail_probability_noncentral_chisquare(mvn, A, t, **kwargs):
        """
        Approximate the upper tail probability using the noncentral chi square approximation.
        Use the scipy implementation of ncx2. This is the method of liu tang zhang.
        https://doi.org/10.1016/j.csda.2009.11.025
        """
        tspecial, dof, noncentrality = compute_ncx2_params(t, mvn.mean, mvn.covariance, A)
        if noncentrality == 0.0:
            # If non-centrality is zero, we can just use the standard chi2 distribution.
            return 1.0 - chi2.cdf(tspecial, dof, loc=0, scale=1)
        else:
            return 1.0 - ncx2.cdf(tspecial, dof , noncentrality)

    @staticmethod
    def upper_tail_probability_imhof(mvn, A, t, **kwargs):
        """ Compute the upper tail probability using the method of imhof. This
            calls this C++ function defined in cpp/imhof.cpp.

        Args:
            t (float): value of t in Prob(Q > t)
            eps_abs (float): absolute error tolerance.
            eps_rel (float): relative error tolerance.
            limit (int, optional): Maximum number of mesh points for numerical integration. Defaults to 1000.

        Returns:
            float: [description]
        """
        eps_abs = kwargs["eps_abs"]
        eps_rel = kwargs["eps_rel"]
        limit = kwargs["limit"]

        # Compute lambda and delta parameters.
        lambdas, deltas = compute_lambdas_deltas(mvn.mean, mvn.covariance, A)

        # Convert to c types.
        t = ctypes.c_double(t)
        lambdas = lambdas.astype(np.float64)
        nlambdas = ctypes.c_int(lambdas.size)
        h = np.ones(lambdas.size, dtype=np.float64)
        delta = deltas.astype(np.float64)
        prob = np.array([0.0], dtype=np.float64)
        epsabs_out = np.array([0.0], dtype=np.float64)
        epsabs = ctypes.c_double(eps_abs)
        epsrel = ctypes.c_double(eps_rel)
        limit = ctypes.c_int(limit)

        # Call the imhof C++ method.
        MvnQuadForm.imhof.upper_tail_prob(t, lambdas, nlambdas, h, delta, prob, epsabs_out, epsabs, epsrel, limit)

        if epsabs_out[0] > 2 * eps_abs:
            warnings.warn("The absolute error is %.3E which is more than double the absolute error tolerance" % eps_abs)
        return prob[0]
    
    @staticmethod
    def compute_moment(mvn, A, t, d):
        """
        Compute the moments of Q(x) - t so that the Chebshev and SOS techniques can be used for
        estimating Prob(Q(x) - t <= 0).
        """
        if d == 0:
            return 1
        elif d == 1:
            # https://en.wikipedia.org/wiki/Quadratic_form_(statistics)
            return np.trace(A @ mvn.covariance) + (mvn.mean.T @ A @ mvn.mean) - t
        elif d == 2:
            # https://en.wikipedia.org/wiki/Quadratic_form_(statistics)
            # Note: variance is invariant under translation, so we don't worry about the -1 component.
            if not check_symmetric(A):
                A = 0.5 * (A + A.T)
            else:
                A = A
            variance = 2 * np.trace(A @ mvn.covariance @ A @ mvn.covariance) + 4 * mvn.mean.T @ A @ mvn.covariance @ A @ mvn.mean
            return variance + MvnQuadForm.compute_moment(mvn, A, t, 1)**2
        elif d > 2:
            raise Warning("Using monte carlo to compute moments.")
            return MvnQuadForm.monte_carlo_moments(mvn, A, t, d)

    @staticmethod
    def monte_carlo_moments(mvn, A, t, d, n_samples = 1e6):
        """
        Estimate the moments of Q(x) - t with Monte Carlo.
        """
        samples = np.random.multivariate_normal(mvn.mean, mvn.covariance, int(n_samples))
        samples = samples.T
        quad_form_samps = (samples.T.dot(A)*samples.T).sum(axis=1)
        quad_form_samps_minus_one = quad_form_samps - t * np.ones(quad_form_samps.shape)
        return np.mean(np.power(quad_form_samps_minus_one, d))


class GmmQuadForm(object):
    """
    This random variable is defined by:
        Q = x'Qx
    Where x is some Gaussian Mixture Model (GMM) and Q is a positive definite matrix.
    """

    @staticmethod
    def upper_tail_probability(gmm, Q, t, method, **kwargs):
        # Choose the method to compute the component probabilities.
        if method == "imhof":
            mvnqf_method = MvnQuadForm.upper_tail_probability_imhof
        elif method == "ltz":
            mvnqf_method = MvnQuadForm.upper_tail_probability_noncentral_chisquare
        elif method == "monte_carlo":
            mvnqf_method = MvnQuadForm.upper_tail_probability_monte_carlo
        else:
            raise Exception("Invalid method input.")

        # Compute the probability as the weighted sum of the components.
        upper_tail_prob = 0
        for component_weight, mvn in gmm:
            mvnqf_prob = mvnqf_method(mvn, Q, t, **kwargs)
            upper_tail_prob += component_weight * mvnqf_prob
        return upper_tail_prob

    @staticmethod
    def compute_moment(gmm, A, t, d):
        moment = 0.0
        for w, mvn in gmm:
            moment += w * MvnQuadForm.compute_moment(mvn, A, t, d)
        return moment