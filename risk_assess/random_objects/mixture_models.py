from risk_assess.random_objects.random_variables import RandomVariable
import numpy as np

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

    def __getitem__(self,index):
         return (self.component_probabilities[index], self.component_random_variables[index])

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
    
    def get_components(self):
        return [(w, rv) for w, rv in zip(self.component_probabilities, self.component_random_variables)]

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