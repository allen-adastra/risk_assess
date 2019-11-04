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

# Declare variables at time "t"
xt = sp.Symbol("x_t")
yt = sp.Symbol("y_t")
vt = sp.Symbol("v_t")
wvt = sp.Symbol("w_{v_t}")
wthetat = sp.Symbol("w_{theta_t}")
thetat = sp.Symbol("theta_t")
cos_thetat = sp.cos(thetat)
sin_thetat = sp.sin(thetat)
sin_thetat = sp.sin(thetat)
cos_thetat = sp.cos(thetat)
sin_wthetat = sp.sin(wthetat)
cos_wthetat = sp.cos(wthetat)

# Create a list of variables at time "t"
# Indicies for the variables:
# 0 : xt
# 1 : yt
# 2 : vt
# 3 : sin_thetat
# 4 : cos_thetat
# 5 : sin_wthetat
# 6 : cos_wthetat

xt_base = BaseVariable(xt, 1,  xt + vt * cos_thetat)
yt_base = BaseVariable(yt, 1, yt + vt * sin_thetat)
vt_base = BaseVariable(vt, 2, vt + wvt)
sin_thetat_base = BaseVariable(sin_thetat, 1000, sp.expand_trig(sp.sin(thetat + wthetat)))
cos_thetat_base = BaseVariable(cos_thetat, 1000, sp.expand_trig(sp.cos(thetat + wthetat)))
sin_wthetat_base = BaseVariable(sin_wthetat, 0, None)
cos_wthetat_base = BaseVariable(cos_wthetat, 0, None)

base_variables = BaseVariables([xt_base, yt_base, vt_base, sin_thetat_base, cos_thetat_base, sin_wthetat_base, cos_wthetat_base])

variable_dependence_graph = nx.Graph()
variable_dependence_graph.add_nodes_from(base_variables.sympy_rep)
variable_dependence_graph.add_edges_from([(xt, yt), (xt, vt), (yt, vt), (xt, sin_thetat), (xt, cos_thetat), (yt, sin_thetat), (yt, cos_thetat)])

# Expressions we have update relations for.
terms_with_generated_relations = [set([xt, sin_thetat]), set([xt, cos_thetat]), set([yt, sin_thetat]), set([yt, cos_thetat])]

# Construct the expression for E[x_{t+1} * y_{t+1}] in terms of expressions at time t
cross = sp.poly(xt_base.update_relation * yt_base.update_relation, base_variables.sympy_rep)
terms_with_generated_relations += [set([xt, yt])]
relations_to_check = [cross]

i = 0

while relations_to_check:
    relations_to_check, terms_with_generated_relations = iterate_relations(relations_to_check, base_variables, variable_dependence_graph, terms_with_generated_relations)