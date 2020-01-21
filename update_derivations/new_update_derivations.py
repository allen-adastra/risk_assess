import sympy as sp
import networkx as nx
import time
import numpy as np

class BaseVariable(object):
    def __init__(self, sympy_rep, max_moment, update_relation):
        self._sympy_rep = sympy_rep
        self._max_moment = max_moment
        self._update_relation = update_relation

    def __str__(self):
        return str(self._sympy_rep)

    def __repr__(self):
        return repr(self._sympy_rep)

    @property
    def max_moment(self):
        return self._max_moment

    @property
    def update_relation(self):
        return self._update_relation
    
    @property
    def sympy_rep(self):
        return self._sympy_rep

    @max_moment.setter
    def max_moment(self, new_max_moment):
        self._max_moment = new_max_moment

class DerivedVariable(object):
    def __init__(self, variable_power_mapping, update_relation):
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
        if variable_power_map == self._variable_power_mapping:
            return True

def generate_higher_degree_update_relation(variable, degree, base_variables):
  """
  Args:
    variable (instance of BaseVariable)
    degree
    base_variables 
  """
  expression = variable.update_relation**degree
  return generate_new_update_relation(expression, base_variables)

def generate_new_update_relation(variable_power_mapping, base_variables):
    """
    Args:
        variable_power_mapping : 
    """
    variable_expansions = [var.update_relation**power for var, power in variable_power_mapping.items()]
    new_relation = np.prod(variable_expansions)

    # If there is only one variable in the variable power mapping, we are essentially
    # increasing the maximum moment that we can compute for that variable.
    if len(variable_power_mapping.keys()) == 1:
        var = list(variable_power_mapping.keys())[0]
        power = list(variable_power_mapping.values())[0]
        #assert power == var.max_moment + 1 # We should not be able to increase the maximum moment by more than one
        var.max_moment = power
    return DerivedVariable(variable_power_mapping, new_relation)

def check_polynomial(poly_to_check, base_variables, dependence_graph, derived_variables):
    """
    Args:
        poly_to_check (sympy polynomial): this polynomial is in the sympy_reps of base_variables with indicies corresponding to base_variables 
        base_variables (list of instances of BaseVariable)
        dependence_graph (networkX graph): nodes consist of set(base_variables)
        derived_variables (list of instances of DerivedVariable)
    """
    components_to_expand = []
    for multi_index in poly_to_check.monoms():
        # Iterate over multi-indicies for the monomials.
        # Construct a dictionary mapping base variables to their power in this monomial.
        variable_power_mapping = {base_variables[i] : multi_index[i] for i in range(len(base_variables))}

        # Find the subgraph induced by multi_index.
        variables_in_mono = {base_variables[i] for i, degree in enumerate(multi_index) if degree != 0}
        mono_dependence_graph = dependence_graph.subgraph(variables_in_mono)
        # IMPORTANT: Both the if and else blocks must generate a set "base_variable_components".
        if len(mono_dependence_graph.edges) == 0:
            # All variables in this monomial are independent of each other.
            # Check that we have correctly found the subgraph.
            assert (len(mono_dependence_graph.nodes) == len(variables_in_mono))
            base_variable_components = variables_in_mono # Both if and else must inclu
        else:
            # Some of the variables in this monomial are dependent on each other, in this case,
            # we need one update relation for each component of the variable dependency graph
            # for this particular monomial.
            connected_components = list(nx.connected_components(mono_dependence_graph))

            # Separate the components into trivial and non-trivial components
            trivial_components = [comp for comp in connected_components if len(comp) == 1]
            nontrivial_components = [comp for comp in connected_components if comp not in trivial_components]
            for component in nontrivial_components:
                # Compute the powers of this component
                component_var_power_map = {var : variable_power_mapping[var] for var in component}
                update_relation_exists = any([derived_var.equivalent_variable_power_mapping(component_var_power_map) for derived_var in derived_variables])
                if (not update_relation_exists) and component_var_power_map not in components_to_expand:
                    # If we don't have an update relation for the component and the component is not already in
                    # the list of components to expand, add it.
                    components_to_expand.append(component_var_power_map)
            base_variable_components = {comp.pop() for comp in trivial_components}

        for var in base_variable_components:
            # If we can't compute the variable to a high enough power, add it to the list of components to expand.
            if variable_power_mapping[var] > var.max_moment:
                components_to_expand.append({var : variable_power_mapping[var]})
    return components_to_expand

def test_iterate(poly_to_check, base_variables, dependence_graph, derived_variables):
    base_variables_sympy = [var.sympy_rep for var in base_variables]
    things_to_expand = check_polynomial(poly_to_check, base_variables, dependence_graph, derived_variables)
    i = 0
    while things_to_expand and i < 100:
        thing = things_to_expand.pop()
        new_derived_variable = generate_new_update_relation(thing, base_variables)
        derived_variables.add(new_derived_variable)
        new_stuff_to_expand = check_polynomial(sp.poly(new_derived_variable.update_relation, base_variables_sympy), base_variables, dependence_graph, derived_variables)
        things_to_expand += new_stuff_to_expand
        i+=1
    print("LEFTOVER")
    print(things_to_expand)
    print("NUmber of derived variables " + str(len(derived_variables)))
    # power_maps = {var.sympy_rep for var in derived_variables}
    # print("NUmber of unique sympy reps " + str(len(power_maps)))
    # for var in derived_variables:
    #     print("Variable: " + str(var.sympy_rep))
    #     print("Expansion: " + str(var.update_relation))
