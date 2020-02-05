import sympy as sp
import numpy as np

class BaseVariable(object):
    """
    A variable of the polynomial system x_{t + 1} = f(x_t).
    """
    def __init__(self, sympy_rep, computable_moments, update_relation):
        """
        Args:
            sympy_rep (Instance of Sp.Symbol): Representation of this variable in SymPy.
            computable_moments (Set of Natural Numbers): Maximum moment that we can compute for this variable.
            update_relation (SymPy expression or None): If this variable is x, then this expression is f(x_t) expanded.
        """
        self._sympy_rep = sympy_rep
        self._computable_moments = computable_moments
        self._update_relation = update_relation

    def __str__(self):
        return str(self._sympy_rep)

    def __repr__(self):
        return repr(self._sympy_rep)

    @property
    def computable_moments(self):
        return self._computable_moments

    @property
    def update_relation(self):
        return self._update_relation
    
    @property
    def sympy_rep(self):
        return self._sympy_rep

    def add_computable_moment(self, n):
        self._computable_moments.add(n)


class DerivedVariable(object):
    def __init__(self, variable_power_mapping, update_relation):
        """
        Args:
            variable_power_mapping (Dictionary Mapping BaseVariable -> Natural Number):
                Every derived variable is of the form x^alpha where x is a vector and alpha is a multi-index.
                variable_power_mapping essentially maps x_i to alpha_i for all i s.t. alpha_i > 0.
            update_relation (SymPy expression): This variable in at time t + 1 as a polynomial in base variables at time t.
        """
        assert min(variable_power_mapping.values()) > 0
        self._variable_power_mapping = variable_power_mapping
        self._update_relation = update_relation

    def __hash__(self):
        return hash(self.sympy_rep)
    
    def __eq__(self, other_obj):
        return hash(self.sympy_rep) == hash(other_obj.sympy_rep)

    @property
    def sympy_rep(self):
        return np.prod([var.sympy_rep**power for var, power in self._variable_power_mapping.items()])

    @property
    def variable_power_mapping(self):
        return self._variable_power_mapping

    @property
    def update_relation(self):
        return self._update_relation

    def equivalent_variable_power_mapping(self, variable_power_map):
        """
        Given another variable power mapping, check if it is equal to that of this variable.
        Args:
            variable_power_mapping (Dictionary Mapping BaseVariable -> Natural Number): variable power mapping to check.
        """
        if variable_power_map == self._variable_power_mapping:
            return True