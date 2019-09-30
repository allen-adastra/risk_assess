from scipy.special import comb

class CharacteristicFunction(object):
    def __init__(self):
        pass

    def compute_value(self, t):
        raise NotImplementedError("compute_value not implemented for object of type CharacteristicFunction")


class cBetaCharacteristicFunction(object):
    def __init__(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def compute_value(self, t):
        