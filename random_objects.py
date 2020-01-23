from itertools import accumulate
import numpy as np
import math
import cmath
from scipy.special import hyp1f1, comb
from scipy.stats import norm, ncx2, chi2
import scipy.io

class RandomVariable(object):
    def __init__(self):
        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    """
    Virtual methods.
    """
    def compute_moment(self, order):
        raise NotImplementedError("Method compute_moments() is not implemented.")

    def sample(self):
        raise NotImplementedError("Method sample() is not implemented.")

    def compute_characteristic_function(self, t):
        raise NotImplementedError("Method compute_characteristic_function() is not implemented.")
    
    """
    Implemented methods.
    """
    def compute_moments(self, order):
        return [self.compute_moment(i) for i in range(order + 1)]

    def compute_variance(self):
        return self.compute_moment(2) - self.compute_moment(1)**2


class RandomVector(object):
    def __init__(self, random_variables):
        # random_variables: list of random variables
        self.random_variables = random_variables

    def compute_vector_moments(self, n_moments):
        # n_moments is a list of numbers of the maximum moment order to be computed for each random variable
        all_moments = len(n_moments) * [None] #list of lists
        for i in range(len(n_moments)):
            all_moments[i] = self.random_variables[i].compute_moments(n_moments[i])
        return all_moments

    def sample(self):
        return [var.sample() for var in self.random_variables]

    def dimension(self):
        return len(self.random_variables)

"""
Finite State Time-Homogenous Markov Chain. 
"""
class MarkovChain(object):
    def __init__(self, T, p0):
        """
        T: n x n transition matrix
        p0: n x 1 vector of transition probabilities
        """
        self.T = T
        self.p0 = p0

        # Dictionary mapping time step "i" to a vector of the
        # marginal probabilities at time "i"
        self.marginal_probabilities = {0 : p0}

    def get_marginal_vector(self, n_step):
        """
        Get the vector of marginal probabilities at time n_step.
        """
        assert isinstance(n_step, int) == True
        assert n_step >= 0
        if n_step not in self.marginal_probabilities.keys():
            # In this case, we don't have the desired marginal probability vector yet.

            # Identify the highest time step for which we have 
            # the vector of marginal probabilities
            max_i = max(self.marginal_probabilities.keys())

            # Propagate marginal probabilities forward in time.
            for i in range(max_i, n_step):
                # The vector of marginal probabilities at time t + 1 is the transition matrix multiplied
                # by the vector of marginal probabilities at time t
                self.marginal_probabilities[i + 1] = np.matmul(self.T, self.marginal_probabilities[i])
        return self.marginal_probabilities[n_step]    

"""
A sequence of Gaussian Mixture Models (GMMs) that represents predicted controls.
The weights are invariant across time.
"""
class GmmControlSequence(object):
    def __init__(self, gmms, dt, max_weight_error_tolerance= 1e-6 ):
        """
        Args:
            gmms (list of instances of GMMs): For every bivariate normal, the first one is accel and second is steer
        """
        self._gmms = gmms
        self._n_components = len(self._gmms[0].component_random_variables)
        self.check_consistency()
        self._weights = self._gmms[0].component_probabilities # If consistent, all the GMMs will have same component probs

        assert abs(sum(self._weights) - 1) < max_weight_error_tolerance
        self.generate_rv_array_rep()

    @property
    def array_rep(self):
        return self._array_rv_rep

    @classmethod
    def from_prediction(cls, prediction, dt):
        """
        Convert a prediction output from the deep net into an instance of GmmControlSequence.
        Scale the parameters from the deep net according to the time step.
        Args:
            prediction (output of PyTorch deep net)
            dt (scalar) : seconds in between each time step
        """
        gmms = []
        for tstep, pre in enumerate(prediction):
            weights_accel = np.array(pre['lweights_acc'][0].exp().tolist())
            weights_alpha = np.array(pre['lweights_alpha'][0].exp().tolist())
            mus_accel = np.array(pre['mus_acc'][0].tolist())
            sigs_accel = np.array(pre['lsigs_acc'][0].exp().tolist())
            mus_alpha = np.array(pre['mus_alpha'][0].tolist())
            sigs_alpha = np.array(pre['lsigs_alpha'][0].exp().tolist())

            n_accel_modes = len(weights_accel)
            n_alpha_modes = len(weights_alpha)

            mixture_components = (n_accel_modes * n_alpha_modes) * [None]
            # Assume accel and steer modes are independent of each other, so there are len(weights_accel) * len(weights_alpha) modes
            for i in range(n_accel_modes):
                for j in range(n_alpha_modes):
                    w = weights_accel[i] * weights_alpha[j]
                    mu = dt * np.array([[mus_accel[i]], [mus_alpha[j]]])
                    cov = (dt**2) * np.array([[sigs_accel[i]**2, 0], [0, sigs_alpha[j]**2]])
                    mn = MultivariateNormal(mu, cov)
                    mixture_components[i * n_alpha_modes + j] = (w, mn)
            gmms.append(GMM(mixture_components))
        gmm_control_seq = cls(gmms, dt)
        return gmm_control_seq

    def generate_rv_array_rep(self):
        """
        Instead of a list of GMMs, we have two arrays of instance of the class Normal.
        Each row of an array corresponds to the mode.
        """
        # Preallocate memory
        accel_rvs = [len(self._gmms) * [None] for i in range(self._n_components)]
        steer_rvs = [len(self._gmms) * [None] for i in range(self._n_components)]
        for i in range(self._n_components):
            for j in range(len(self._gmms)):
                mvn = self._gmms[j].component_random_variables[i]
                accel_rvs[i][j] = Normal(mvn.mean[0][0], mvn.covariance[0][0])
                steer_rvs[i][j] = Normal(mvn.mean[1][0], mvn.covariance[1][1])
        self._array_rv_rep = {"accels" : accel_rvs, "steers" : steer_rvs, "weights" : self._weights}

    def check_consistency(self):
        # First check that the number of components for each GMM is the same.
        components_per_gmm = [len(gmm.component_random_variables) for gmm in self._gmms]
        assert len(set(components_per_gmm)) == 1

        # Check that the weights of the components are consistent across time.
        for i in range(self._n_components):
            comp_weights = [gmm.component_probabilities[i] for gmm in self._gmms]
            assert(len(set(comp_weights))) == 1
    
    def sample(self, n_samples):
        """
        Draw sample control sequences.
        Args:
            n_samples
        Returns:
            accel_samps (n_samples x sequence length numpy array): Each row is a sample control sequence
            steer_samps (n_samples x sequence length numpy array): Each row is a sample control sequence
        """
        # Determine the number of samples from each mode
        mode_idx = [list(foo).index(1) for foo in np.random.multinomial(1, self._weights, size = n_samples)]
        n_samps_per_mode = [mode_idx.count(i) for i in range(self._n_components)]

        # Arrays to hold samples from separate modes
        accel_arrays = self._n_components * [None]
        steer_arrays = self._n_components * [None]
        for i in range(self._n_components):
            # Sample each mode.
            n_samps = n_samps_per_mode[i]

            # Each row of these two arrays will contain a sample contrl sequence
            mode_accels = np.zeros((n_samps, len(self._gmms)))
            mode_steers = np.zeros((n_samps, len(self._gmms)))

            # Sample the relevant multivariate normals and fill out the arrays
            for j in range(len(self._gmms)):
                foo = self._gmms[j].component_random_variables[i].sample(n_samps)
                mode_accels[:, j] = foo[:, 0]
                mode_steers[:, j] = foo[:, 1]
            accel_arrays[i] = mode_accels
            steer_arrays[i] = mode_steers
        accel_samps = np.vstack(accel_arrays)
        steer_samps = np.vstack(steer_arrays)
        return accel_samps, steer_samps

"""
A sequence of Gaussian Mixture Models (GMMs) that represent a predicted agent trajectory.
"""
class GmmTrajectory(object):
    def __init__(self, gmms):
        """
        Args:
            gmms (list of instance of MixtureModel): ordered list of Gaussian Mixture Models representing a predicted agent trajectory.
        """
        self._gmms = gmms
        self._n_components = len(self._gmms[0].component_random_variables)
        self._n_steps = len(self._gmms)
        self._dimension = self._gmms[0].component_random_variables[0].dimension
        self.check_consistency()
        self.generate_array_rep()
    
    def __len__(self):
        return len(self._gmms)

    @property
    def array_rep(self):
        self.generate_array_rep()
        return self._mean_trajectories, self._covariance_trajectories, self._weights
    
    @property
    def gmms(self):
        return self._gmms

    @classmethod
    def from_prediction(cls, prediction, scale_k):
        """
        Convert a predicted GMM trajectory into an instance of GmmTrajectory.
        Args:
            prediction: output from Pytorch deep net
            scale_k: position down scaling factor
        """
        gmms = []
        for pre in prediction:
            weights = np.array(pre['lweights'][0].exp().tolist())
            # transform mus and sigs to global frame
            mus = np.array(pre['mus'][0].tolist())
            sigs = np.array(pre['lsigs'][0].exp().tolist())

            # TODO: The matrix cov rows correspond to components and ultimately x and y are uncorrelated
            num_mixture = mus.shape[0]
            mixture_components = num_mixture * [None] # List of tuples of the form (weight, MultivariateNormal)
            for k in range(num_mixture):
                # get covariance matrix in local frame
                cov_k = np.array([[sigs[k,0]**2,0],[0,sigs[k,1]**2]])
                mu = np.c_[mus[k]] # convert mus[k] which is a list into a column numpy array
                mn = MultivariateNormal(mu*scale_k, cov_k*scale_k)

                # Add to mixture_components
                mixture_components[k] = (weights[k], mn)
            gmms.append(GMM(mixture_components))
        gmm_traj = cls(gmms)
        return gmm_traj

    def check_consistency(self):
        # First check that the number of components for each GMM is the same.
        components_per_gmm = [len(gmm.component_random_variables) for gmm in self._gmms]
        assert len(set(components_per_gmm)) == 1

        # Check that the weights of the components are consistent across time.
        for i in range(self._n_components):
            comp_weights = [gmm.component_probabilities[i] for gmm in self._gmms]
            assert(len(set(comp_weights))) == 1

    def generate_array_rep(self):
        """
        Generate lists of trajectories for mean and covariances
        """
        self._mean_trajectories = self._n_components * [None]
        self._covariance_trajectories = self._n_components * [None]
        self._weights = self._gmms[0].component_probabilities # We can use the weights of the first gmm as check_consistency() ensures they are all the same
        for i in range(self._n_components):
            # Generate the sequence of mean and covariances for the ith mode
            means = np.zeros((self._n_steps, self._dimension))
            covs = np.zeros((self._n_steps, self._dimension, self._dimension))
            for j, gmm in enumerate(self._gmms):
                # We are currently looking at the ith mode
                rv = gmm.component_random_variables[i]
                means[j, :] = rv.mean.T
                covs[j] = rv.covariance
            self._mean_trajectories[i] = means
            self._covariance_trajectories[i] = covs
    
    def change_frame(self, offset_vec, rotation_matrix):
        """
        Change from frame A to frame B.
        Args:
            offset_vec (nx1 numpy array): vector from origin of frame A to frame B
            rotation_matrix (n x n numpy array): rotation matrix corresponding to the angle of the x axis of frame A to frame B
        """
        # Apply to each component of each GMM in the trajectory.
        for gmm in self._gmms:
            gmm.change_frame(offset_vec, rotation_matrix)

    def save_as_matfile(self, directory, filename):
        """
        Save parameters of this GMM trajectory as a mat file
        """
        if ".mat" not in filename:
            filename = filename + ".mat"
        if directory[-1] != "/":
            directory = directory + "/"
        fullpath = directory + filename
        matfile_dic = {}
        for i in range(self._n_components):
            comp_key = "component_" + str(i)
            mean_traj = self._mean_trajectories[i]
            cov_traj = self._covariance_trajectories[i]
            n_steps = len(mean_traj)
            mean_array = np.zeros((2, n_steps))
            cov_array = np.zeros((2, 2, n_steps))
            for j in range(n_steps):
                mean_array[:, j] = mean_traj[j]
                cov_array[:, :, j] = cov_traj[j]
            matfile_dic[comp_key] = {"means" : mean_array, "covariances" : cov_array, "weight" : self._weights[i]}
        scipy.io.savemat(fullpath, matfile_dic)

class MixtureModel(RandomVariable):
    def __init__(self, mixture_components, weight_tolerance = 1e-6):
        """
        component_random_variables: list of tuples of the form (weight, RandomVariable)
        """
        self.component_probabilities = [comp[0] for comp in mixture_components]
        self.component_random_variables = [comp[1] for comp in mixture_components]
        sum_probs = sum(self.component_probabilities)
        if abs(sum_probs - 1) > weight_tolerance:
            raise Exception("Input component probabilities must sum to within " + str(weight_tolerance) + " of 1, but it sums to: " + str(sum_probs))
        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            moment = 0
            for rv, prob in zip(self.component_random_variables, self.component_probabilities):
                moment += prob * rv.compute_moment(order)
            self._moment_values[order] = moment
        return self._moment_values[order]

    def compute_characteristic_function(self, t):
        if t not in self._char_fun_values.keys():
            char_fun = 0
            for rv, prob in zip(self.component_random_variables, self.component_probabilities):
                char_fun += prob * rv.compute_characteristic_function(t)
            self._char_fun_values[t] = char_fun
        return self._char_fun_values[t]

    def sample(self):
        # Draw one sample from the multinomial distribution
        # np.random.multinomial(1, [0.2, 0.8]) for example will return either
        # array([0, 1]) or array([1, 0]) where the index of the 1 corresponds to
        # the variable chosen.
        mode_idx = list(np.random.multinomial(1, self.component_probabilities)).index(1)
        return self.component_random_variables[mode_idx].sample()

class GMM(MixtureModel):
    """
    Multivariate Gaussian Mixture Model (GMM)
    """

    def change_frame(self, offset_vec, rotation_matrix):
        """
        Change from frame A to frame B.
        Args:
            offset_vec (nx1 numpy array): vector from origin of frame A to frame B
            rotation_matrix (n x n numpy array): rotation matrix corresponding to the angle of the x axis of frame A to frame B
        """
        for mvn in self.component_random_variables:
            mvn.change_frame(offset_vec, rotation_matrix)

class MultivariateNormal(object):
    def __init__(self, mean, covariance):
        """
        Args:
            mean (n x 1 numpy array): mean vector
            covariance (n x n numpy array): covariance matrix
        """
        self._mean = mean
        self._covariance = covariance

    @property
    def mean(self):
        return self._mean
    
    @property
    def covariance(self):
        return self._covariance

    @property
    def dimension(self):
        return self._mean.shape[0]

    def sample(self, n_samps):
        return np.random.multivariate_normal(self._mean.flatten(), self._covariance, int(n_samps))

    def rotate(self, rotation_matrix):
        """
        Express the MVN in a new frame that is rotated by the rotation matrix
        Args:
            dtheta (rotation_matrix):
        """
        # For counter clockwise rotations, need to multiply dtheta by -1.
        self._mean = np.matmul(rotation_matrix, self._mean)
        self._covariance = np.matmul(rotation_matrix, np.matmul(self._covariance, rotation_matrix.T))

    def change_frame(self, offset_vec, rotation_matrix):
        """
        Change from frame A to frame B.
        Args:
            offset_vec (nx1 numpy array): vector from origin of frame A to frame B
            rotation_matrix (n x n numpy array): rotation matrix corresponding to the angle of the x axis of frame A to frame B
        """
        # By convention, we need to translate before rotating.
        self._mean += -offset_vec
        self.rotate(rotation_matrix.T) # The inverse of a rotation matrix is equal to its transpose.

class Normal(RandomVariable):
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        self._variance = std**2

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}
    
    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            self._moment_values[order] = norm.moment(order, loc = self._mean, scale = self._std)
        return self._moment_values[order]
    
    def compute_characteristic_function(self, t):
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(-0.5 * self._variance * t**2, self._mean * t))
        return self._char_fun_values[t]
    
    def sample(self):
        return np.random.normal(self._mean, self._std)

class Constant(RandomVariable):
    def __init__(self, value):
        self.value = value

    def compute_moment(self, order):
        if order >= 0:
            return self.value**order
        else:
            raise Exception("Invalid order input")

    def sample(self):
        return self.value

    def compute_characteristic_function(self, t):
        return 1.0

"""cbeta is the beta distribution multiplied by a constant. When c = 1, this
is just a normal beta random variable."""
class cBetaRandomVariable(RandomVariable):
    def __init__(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moments(self, order):
        """
        Compute all beta moments up to the given order. The returned list indices should match the moment orders
        e.g. the return[i] should be the ith beta moment
        """
        fs = map(lambda r: (self.alpha + r)/(self.alpha + self.beta + r), range(order))
        beta = [1] + list(accumulate(fs, lambda prev,n: prev*n))
        cbeta = [beta[i]*self.c**i for i in range(len(beta))]
        return cbeta

    def sample(self):
        return self.c * np.random.beta(self.alpha, self.beta)

    def compute_characteristic_function(self, t):
        """
        The characteristic function of the beta distribution is Kummer's confluent hypergeometric function which 
        is implemented by scipy.special.hyp1f1. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html
        If Phi_X(t) is the characteristic function of X, then for any constant c, the characteristic function of cX is
        Phi_cX(t) = Phi_X(ct)
        """
        return hyp1f1(self.alpha, self.beta, self.c * t)

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
This object is the random variable c + x1 + x2 + ... + xn
"""
class SumOfRVs(object):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

    def cos_applied(self):
        return CosSumOfRVs(self.c, self.random_variables)

    def sin_applied(self):
        return SinSumOfRVs(self.c, self.random_variables)

    def add_rv(self, rv):
        self.random_variables.append(rv)


"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
This object is the random variable cos(c + x1 + x2 + ... + xn)
"""
class CosSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}
        
    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            if order == 1:
                return np.real(self.compute_characteristic_function(1))
            elif order > 1:
                n = math.floor(order/2)
                # The ith element of real_component is the real component of CharacteristicFunction(i)
                char_fun_values = [self.compute_characteristic_function(i) for i in range(order + 1)]
                real_component = [np.real(val) for val in char_fun_values]
                # Different expressions depending on if the order is odd or even
                if order % 2 == 0:
                    summation = sum([comb(order, k) * real_component[2 * (n - k)] for k in range(n)])
                    self._moment_values[order] = (1.0/(2.0**(2.0 * n))) * comb(order, n) + (1.0/(2.0**(2.0 * n - 1))) * summation
                elif order % 2 == 1:
                    summation = sum([comb(2 * n + 1, k) * real_component[2 * n + 1 - 2 * k] for k in range(n + 1)])
                    self._moment_values[order] = (1.0/(4.0**n)) * summation
                else:
                    raise Exception("Input order mod 2 is neither 0 nor 1")
            else:
                raise Exception("Input order must be positive.")
        return self._moment_values[order]

    def compute_moments(self, order):
            return [self.compute_moment(i) for i in range(order + 1)]

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, t * self.c)) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]
    
    def add_rv(self, rv):
        self.random_variables.append(rv)
    
    def add_constant(self, c):
        self.c += c

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
This object is the random variable sin(c + x1 + x2 + ... + xn)
"""
class SinSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            if order == 1:
                return np.imag(self.compute_characteristic_function(1))
            elif order > 1:
                n = math.floor(order/2)
                # The ith element of real_component is the real component of CharacteristicFunction(i)
                char_fun_values = [self.compute_characteristic_function(i) for i in range(order + 1)]
                real_component = [np.real(val) for val in char_fun_values]
                imaginary_component = [np.imag(val) for val in char_fun_values]
                # Different expressions depending on if the order is odd or even
                if order % 2 == 0:
                    summation = sum([((-1)**k) * comb(order, k) * real_component[2 * (n - k)] for k in range(n)])
                    self._moment_values[order] = (1.0/(2.0**(2.0 * n))) * comb(order, n) + (((-1)**n)/ (2**(2 * n - 1))) * summation
                elif order % 2 == 1:
                    summation = sum([((-1)**k) * comb(order, k) * imaginary_component[2 * n + 1 - 2 * k] for k in range(n+1)])
                    self._moment_values[order] = (((-1)**n)/(4**n)) * summation
                else:
                    raise Exception("Input order mod 2 is neither 0 nor 1")
            else:
                raise Exception("Input order must be positive.")
        return self._moment_values[order]

    def compute_moments(self, order):
        return [self.compute_moment(i) for i in range(order + 1)]

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, 1) * t * self.c) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]

    def add_rv(self, rv):
        self.random_variables.append(rv)
    
    def add_constant(self, c):
        self.c += c

"""
Let x1, x2, ..., xn be n independent random variables and c be some constant.
Let theta = c + x1 + x2 + ... + xn
This object is the random variable cos(theta) * sin(theta)
"""
class CrossSumOfRVs(RandomVariable):
    def __init__(self, c, random_variables):
        self.c = c
        self.random_variables = random_variables
        self.cos_theta = CosSumOfRVs(c, random_variables)
        self.sin_theta = SinSumOfRVs(c, random_variables)

        # Cached values.
        self._char_fun_values = {}
        self._moment_values = {}

    def compute_moment(self, order):
        if order not in self._moment_values.keys():
            if order == 1:
                self._moment_values[order] = 0.5 * np.imag(self.compute_characteristic_function(2))
            else:
                raise Exception("Input order is not currently supported.")
        return self._moment_values[order]

    def compute_covariance(self):
        return self.compute_moment(1) - self.cos_theta.compute_moment(1) * self.sin_theta.compute_moment(1)

    def compute_characteristic_function(self, t):
        """
        If there are n random variables in self.random_variables w_i for i = 1,...,n
        And we have some constant c, then this function computes the characteristic function of
        c + w_1 + ... + w_n
        """
        if t not in self._char_fun_values.keys():
            self._char_fun_values[t] = cmath.exp(complex(0, 1) * t * self.c) * np.prod([rv.compute_characteristic_function(t) for rv in self.random_variables])
        return self._char_fun_values[t]
