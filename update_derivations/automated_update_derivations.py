import sympy as sp
import networkx as nx
import time

class BaseVariables(object):
    def __init__(self, base_variables):
        """
        Args:
            base_variables : list of base_variables
        """
        self._base_variables = base_variables

    @property
    def sympy_rep(self):
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
    def max_moment(self):
        return self._max_moment

    @property
    def update_relation(self):
        return self._update_relation
    

def identify_needed_updates(polynomial, base_variables, dependence_graph, terms_with_generated_relations):
    """
    Args:
        polynomial: sympy polynomial in the base variables
        base_variables: instance of BaseVariables
        dependence_graph: 
        terms_with_generated_relations: 
    """
    new_update_relations_needed = []
    variables_need_higher_moments = dict()
    for mono in polynomial.monoms():
        for i, degree in enumerate(mono):
            # Check if we can compute moments of the random variable up to "degree"
            # If not, we will need to derive an update relation for the "degree-th"
            # moment of the ith base variable.
            if degree > base_variables.max_moments[i]:
                variables_need_higher_moments[base_variables.variables[i]] = degree
        # Find the graph of variable dependencies for this particular monomial.
        variables_in_mono = {base_variables.sympy_rep[i] for i, degree in enumerate(mono) if degree != 0}
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
                update_relation_exists = any([component == group for group in terms_with_generated_relations]) # Does an update relation already exist for this component?
                if ((component_non_trivial) and (not update_relation_exists) and (component not in new_update_relations_needed)):
                    # DOUBLE CHECK.
                    new_update_relations_needed.append(component)
    return new_update_relations_needed, variables_need_higher_moments

def generate_new_update_relations(need_update_relations, base_variables):
    """
    Args:
        need_update_relations: list of frozensets of variables
        base_variables: 
    Returns:
        new_update_relations
    """
    new_update_relations = []
    variable_base_variable_map = {}
    for var in base_variables.variables:
        variable_base_variable_map[var.sympy_rep] = var
    for new_relation_vars in need_update_relations:
        new_relation = 1
        for sympy_var in new_relation_vars:
            new_relation *= variable_base_variable_map[sympy_var].update_relation
        new_update_relations.append(sp.poly(new_relation, base_variables.sympy_rep))
    return new_update_relations

def iterate_relations(relations_to_check, base_variables, variable_dependence_graph, terms_with_generated_relations):
    """
    Args:
        relations_to_check: 
        base_variables: 
        variable_dependence_graph: 
        terms_with_generated_relations: 
    Returns:
        new_relations_to_check
        terms_with_generated_relations
    """
    need_update_relations = []
    # Given a list of relations to check, generate a new set of relations to check.
    for relation in relations_to_check:
        new_update_relations_needed, variables_need_higher_moments = identify_needed_updates(relation, base_variables, variable_dependence_graph, terms_with_generated_relations)
        need_update_relations += new_update_relations_needed

    # Eliminate redundant stuff
    need_update_relations = {frozenset(s) for s in need_update_relations}
    need_update_relations = [set(s) for s in need_update_relations]
    
    # Generate new relations based off what we have in need_update_relations
    new_relations_to_check = generate_new_update_relations(need_update_relations, base_variables)
    terms_with_generated_relations += need_update_relations
    time.sleep(0.5)
    return new_relations_to_check, terms_with_generated_relations