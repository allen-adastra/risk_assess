import sympy as sp
import networkx as nx
import time

class UpdateDeriver(object):
    def __init__(self, base_variables, dependence_graph):
        """
        Args:
            base_variables: instances of BaseVariables
            dependence_graph: networkx graph with nodes as instances of BaseVariable and edges between two variables if they are dependent
        """
        self._base_variables = base_variables
        self._variable_dependence_graph = dependence_graph


class BaseVariables(object):
    def __init__(self, base_variables):
        """
        Args:
            base_variables : list of base_variables
        """
        self._base_variables = base_variables

    @property
    def sympy_reps(self):
        return [var.sympy_rep for var in self._base_variables]

    @property
    def variables(self):
        return self._base_variables

    @property
    def max_moments(self):
        return [var.max_moment for var in self._base_variables]

    def add_base_variable(self, var):
        self._base_variables.append(var)

class BaseVariable(object):
    def __init__(self, sympy_symbol, max_moment, update_relation):
        """
        Args:
            sympy_symbol: sympy symbolic representation of this variable
            max_moment: maximum moment that we currently can compute for this variable
            update_relation: how to determine this variable at time t + 1 in terms of other the sympy representation of other base variables at time t
        """
        self._sympy_symbol = sympy_symbol
        self._max_moment = max_moment
        self._update_relation = update_relation

    @property
    def sympy_rep(self):
        return self._sympy_symbol

    @property
    def string_rep(self):
        return str(self._sympy_symbol)
    
    @property
    def max_moment(self):
        return self._max_moment

    @property
    def update_relation(self):
        """
        NTS: Should be an instance of a sympy polynomial.
        """
        return self._update_relation


"""
A "DerivedVariable" is a BaseVariable that is created from the product of other base variables.
"""
class DerivedVariable(BaseVariable):
    def __init__(self, component_variables, update_relation):
        """
        Args:
            component_variables: list of instances of BaseVariable s.t. the product of the variables in it form this variable.
            update_relation: 
        """
        variable_string_reps = [str(var.sympy_rep) for var in component_variables]
        max_moment = 1 # By default, we only know how to compute the moments up to 1 of DerivedVariable
        super(DerivedVariable, self).__init__(sp.Symbol(' * '.join(variable_string_reps)), max_moment, update_relation)
        self._component_variables = list(component_variables)

    def __str__(self):
        return str({var.sympy_rep for var in self.component_variables})

    @property
    def component_variables(self):
        return self._component_variables

    @property
    def component_variables_sympy(self):
        return [var.sympy_rep for var in self._component_variables]

    def equivalent_variables(self, set_of_base_variables):
        """
        set_of_base_variables: is the product of the variables in set_of_base_variables equivalent to this variable?
        """
        return set(self.component_variables) == set_of_base_variables

    def sympy_equivalent(self, set_of_sympy_vars):
        """
        set_of_sympy_vars: is the product of the sympy vars in this set equivalent to this variable?
        """
        return set_of_sympy_vars == {var.sympy_rep for var in self.component_variables}


def identify_needed_updates(derived_variable_to_check, base_variables, dependence_graph, derived_variables):
    """
    Args:
        derived_variable_to_check: instance of DerivedVariable
        base_variables: instance of BaseVariables
        dependence_graph: networkx with nodes as variables in BaseVariables and edges representing the two variables are dependent
        derived_variables: list of instances of DerivedVariable, should include derived_variable_to_check
    Returns:
        new_update_relations_needed: list of sets of sympy variables tha
    """
    assert derived_variable_to_check in derived_variables
    new_update_relations_needed = []
    variables_need_higher_moments = dict()
    for mono in derived_variable_to_check.update_relation.monoms():
        for i, degree in enumerate(mono):
            # Check if we can compute moments of the random variable up to "degree"
            # If not, we will need to derive an update relation for the "degree-th"
            # moment of the ith base variable.
            if degree > base_variables.max_moments[i]:
                variables_need_higher_moments[base_variables.variables[i]] = degree
        # Find the graph of variable dependencies for this particular monomial.
        variables_in_mono = {base_variables.variables[i] for i, degree in enumerate(mono) if degree != 0}
        mono_dependence_graph = dependence_graph.subgraph(variables_in_mono)
        if len(mono_dependence_graph.edges) == 0:
            # All variables in this monomial are independent of each other.
            # Check that we have correctly found the subgraph.
            assert (len(mono_dependence_graph.nodes) == len(variables_in_mono))
        else:
            # Some of the variables in this monomial are dependent on each other, in this case,
            # we need one update relation for each component of the variable dependency graph
            # for this particular monomial.
            connected_components = nx.connected_components(mono_dependence_graph)
            for component in connected_components:
                component_non_trivial = len(component) > 1 # If there is only one node in the component, it is trivial.
                update_relation_exists = any([derived_var.equivalent_variables(component) for derived_var in derived_variables]) # Does an update relation already exist for this component?
                if ((component_non_trivial) and (not update_relation_exists) and (component not in new_update_relations_needed)):
                    new_update_relations_needed.append(component)
    return new_update_relations_needed, variables_need_higher_moments

def generate_new_update_relations(need_update_relations, base_variables):
    """
    Args:
        need_update_relations: list of sets of instances of BaseVariable.
        base_variables: 
    Returns:
        new_derived_vars
    """
    new_derived_vars = []
    for list_of_vars in need_update_relations:
        # Take the product of update relations of variables in list_of_vars.
        new_relation = 1
        for var in list_of_vars:
            new_relation *= var.update_relation
        # Make the new relation a sympy polynomial
        new_relation = sp.poly(new_relation, base_variables.sympy_reps)
        new_derived_vars.append(DerivedVariable(list_of_vars, new_relation))
    return new_derived_vars

def iterate_relations(derived_base_vars_to_check, base_variables, variable_dependence_graph, derived_base_vars):
    """
    Args:
        derived_base_vars_to_check: 
        base_variables: 
        variable_dependence_graph: 
        terms_with_generated_relations: 
    Returns:
        new_derived_base_vars
        derived_base_vars
    """
    need_update_relations = []
    # Given a list of relations to check, generate a new set of relations to check.
    for relation in derived_base_vars_to_check:
        new_update_relations_needed, variables_need_higher_moments = identify_needed_updates(relation, base_variables, variable_dependence_graph, derived_base_vars)
        need_update_relations += new_update_relations_needed

    if len(variables_need_higher_moments):
        raise Exception("There were variables that need higher moments, but we haven't figured out yet what to do when certain variables need higher moments.")

    # Generate new relations based off what we have in need_update_relations
    new_derived_base_vars = generate_new_update_relations(need_update_relations, base_variables)
    derived_base_vars += new_derived_base_vars
    return new_derived_base_vars, derived_base_vars