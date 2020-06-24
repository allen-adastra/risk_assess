import numpy as np
from risk_assess.random_objects.mixture_models import GMM
from risk_assess.random_objects.multivariate_normal import MultivariateNormal, Normal
"""
A sequence of Gaussian Mixture Models (GMMs) that represents predicted controls.
The weights are invariant across time.
"""
class GmmControlSequence(object):
    def __init__(self, gmms, max_weight_error_tolerance= 1e-6 ):
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
    def from_prediction(cls, prediction):
        """
        Convert a prediction output from the deep net into an instance of GmmControlSequence.
        Scale the parameters from the deep net according to the time step.
        Args:
            prediction (output of PyTorch deep net)
            dt (scalar) : seconds in between each time step
        """
        gmms = []
        for pre in prediction:
            # Iterate over time steps.
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
                    mu = np.array([[mus_accel[i]], [mus_alpha[j]]])
                    cov = np.array([[sigs_accel[i]**2, 0], [0, sigs_alpha[j]**2]])
                    mn = MultivariateNormal(mu, cov)
                    mixture_components[i * n_alpha_modes + j] = (w, mn)
            gmms.append(GMM(mixture_components))
        gmm_control_seq = cls(gmms)
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
                accel_rvs[i][j] = Normal(mvn.mean[0][0], mvn.covariance[0][0]**0.5)
                steer_rvs[i][j] = Normal(mvn.mean[1][0], mvn.covariance[1][1]**0.5)
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
        assert sum(n_samps_per_mode) == n_samples

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
                # Sample the 
                mvn_sample = self._gmms[j].component_random_variables[i].sample(n_samps)
                mode_accels[:, j] = mvn_sample[:, 0]
                mode_steers[:, j] = mvn_sample[:, 1]
            accel_arrays[i] = mode_accels
            steer_arrays[i] = mode_steers
        accel_samps = np.vstack(accel_arrays)
        steer_samps = np.vstack(steer_arrays)
        return accel_samps, steer_samps