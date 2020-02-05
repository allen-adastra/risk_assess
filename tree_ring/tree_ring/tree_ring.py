import sympy as sp
import networkx as nx
import numpy as np
from tree_ring.objects import BaseVariable, DerivedVariable

def tree_ring(var_power_map, base_variables, dependence_graph, derived_variables, max_iters = 100):
    """
    NOTE (02/04/2020): Currently, this is not implemented recursively as described in the RSS paper.
    Args:
        poly_to_check (Sympy variable in)
        base_variables (Set of instances of BaseVariable)
        dependence_graph (NetworkX Graph)
        derived_variables (Set of instances of DerivedVariable)
    """
    var_power_maps_to_expand = [var_power_map]
    base_variables_sympy = [var.sympy_rep for var in base_variables]
    i = 0 # Keep track of iteratios 
    while var_power_maps_to_expand and i < max_iters:
        new_map_to_expand = var_power_maps_to_expand.pop()
        new_derived_variable = generate_new_update_relation(new_map_to_expand, base_variables)
        derived_variables.add(new_derived_variable)
        new_var_power_maps = check_polynomial(sp.poly(new_derived_variable.update_relation, base_variables_sympy), base_variables, dependence_graph, derived_variables)
        var_power_maps_to_expand += new_var_power_maps
        i+=1
    
    if i >= max_iters:
        raise Exception("Exceeded maximum number of iterations.")

def generate_new_update_relation(variable_power_mapping, base_variables):
    """
    Args:
        variable_power_mapping (Dictionary Mapping BaseVariable -> Natural Number): mapping of base variables with non-zero
            power to their powers.
        base_variables (List of instances of BaseVariable): base variables of the polynomial system. 
    """
    # All base variables in the mapping should have non-zero power.
    #assert min(variable_power_mapping.values()) > 0
    variable_power_mapping = {var : power for var, power in variable_power_mapping.items() if power != 0}

    # Derive the new relation.
    variable_expansions = [var.update_relation**power for var, power in variable_power_mapping.items()]
    new_relation = np.prod(variable_expansions)

    # If there is only one variable in the variable power mapping, we are essentially
    # increasing the maximum moment that we can compute for that variable.
    if len(variable_power_mapping.keys()) == 1:
        var = list(variable_power_mapping.keys())[0]
        power = list(variable_power_mapping.values())[0]
        var.add_computable_moment(power)
    return DerivedVariable(variable_power_mapping, new_relation)

def check_polynomial(poly_to_check, base_variables, dependence_graph, derived_variables):
    """
    Args:
        poly_to_check (sympy polynomial): this polynomial is in the sympy_reps of base_variables with indicies corresponding to base_variables 
        base_variables (list of instances of BaseVariable): 
        dependence_graph (networkX graph): This is the Variable Dependency Graph; its nodes consist of base_variables
        derived_variables (list of instances of DerivedVariable): All of the instances of DerivedVariable created thus far.
    Returns:

    """
    new_variables_to_derive = []
    for multi_index in poly_to_check.monoms():
        # Iterate over multi-indicies for the monomials.
        # Construct a dictionary mapping base variables to their power in this monomial.
        variable_power_mapping = {base_variables[i] : multi_index[i] for i in range(len(base_variables))}

        # Find the subgraph induced by multi_index.
        variables_in_mono = {base_variables[i] for i, degree in enumerate(multi_index) if degree != 0}

        # Find the subgraphed induced by multi_index.
        mono_dependence_graph = dependence_graph.subgraph(variables_in_mono)

        # IMPORTANT: Both the if and else blocks must generate a set "base_variable_components".
        if len(mono_dependence_graph.edges) == 0:
            # All variables in this monomial are independent of each other.
            base_variable_components = variables_in_mono
        else:
            # Some of the variables in this monomial are dependent on each other, in this case,
            # we need one update relation for each component of the variable dependency graph
            # for this particular monomial.
            connected_components = list(nx.connected_components(mono_dependence_graph))

            # Separate the components into trivial and non-trivial components.
            trivial_components = [comp for comp in connected_components if len(comp) == 1]
            nontrivial_components = [comp for comp in connected_components if comp not in trivial_components]

            # Iterate over non-trivial components.
            for component in nontrivial_components:
                # Compute the variable power map. 
                # Note that variables are only in mono_dependence_graph if they have non-zero degree.
                component_var_power_map = {var : variable_power_mapping[var] for var in component}

                # Check if any DerivedVariable already has the same variable power map as this variable.
                update_relation_exists = any([derived_var.equivalent_variable_power_mapping(component_var_power_map) for derived_var in derived_variables])
                if (not update_relation_exists) and component_var_power_map not in new_variables_to_derive:
                    # If we don't have an update relation for the component and the component is not already in the list of components to expand, add it.
                    new_variables_to_derive.append(component_var_power_map)
            base_variable_components = {comp.pop() for comp in trivial_components}
        
        # Check if we can compute variables to sufficiently high power. If not, add it to the list of variables we need to derive.
        for var in base_variable_components:
            if variable_power_mapping[var] not in var.computable_moments:
                new_variables_to_derive.append({var : variable_power_mapping[var]})
    return new_variables_to_derive