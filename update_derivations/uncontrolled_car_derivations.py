import sympy as sp
import networkx as nx
from automated_update_derivations import *

"""
Setup the variables and update relations for the uncontrolled car problem defined by the following update relations:
    x_{t+1} = x_t + v_t cos(theta_t)
    y_{t+1} = y_t + v_t sin(theta_t)
    v_{t+1} = v_t + w_{v_t}
    theta_{t+1} = theta_t + w_{theta_t}
"""
class UncontrolledCarDerivations(object):
    def __init__(self):
         # Declare variables at time "t"
        xt = sp.Symbol("x_t")
        yt = sp.Symbol("y_t")
        vt = sp.Symbol("v_t")
        wvt = sp.Symbol("w_{v_t}")
        wthetat = sp.Symbol("w_{theta_t}")
        thetat = sp.Symbol("theta_t")
        cos_thetat = sp.cos(thetat)
        sin_thetat = sp.sin(thetat)
        sin_wthetat = sp.sin(wthetat)
        cos_wthetat = sp.cos(wthetat)

        self._xt = BaseVariable(xt, 1,  xt + vt * cos_thetat)
        self._yt = BaseVariable(yt, 1, yt + vt * sin_thetat)
        self._vt = BaseVariable(vt, 2, vt + wvt)
        self._sin_thetat = BaseVariable(sin_thetat, float("inf"), sp.expand_trig(sp.sin(thetat + wthetat)))
        self._cos_thetat = BaseVariable(cos_thetat, float("inf"), sp.expand_trig(sp.cos(thetat + wthetat)))
        self._sin_wthetat = BaseVariable(sin_wthetat, float("inf"), None)
        self._cos_wthetat = BaseVariable(cos_wthetat, float("inf"), None)

        self._base_variables = BaseVariables([self._xt, self._yt, self._vt, self._sin_thetat,
                                            self._cos_thetat, self._sin_wthetat, self._cos_wthetat])

        self._variable_dependence_graph = nx.Graph()
        self._variable_dependence_graph.add_nodes_from(self._base_variables.variables)
        self._variable_dependence_graph.add_edges_from([(self._xt, self._yt), (self._xt, self._vt), (self._yt, self._vt),
                                                        (self._xt, self._sin_thetat), (self._xt, self._cos_thetat), (self._yt, self._sin_thetat),
                                                        (self._yt, self._cos_thetat)])

    def derive_cross(self):
        # From the Fall 2019 ICRA Paper, we have the following update relations:
        derived_base_variables = [DerivedVariable([self._xt, self._sin_thetat], None),
                                  DerivedVariable([self._xt, self._cos_thetat], None),
                                  DerivedVariable([self._yt, self._sin_thetat], None),
                                  DerivedVariable([self._yt, self._cos_thetat], None)]
        # Construct the expression for E[x_{t+1} * y_{t+1}] in terms of expressions at time t
        cross_update = sp.poly(self._xt.update_relation * self._yt.update_relation, self._base_variables.sympy_reps)
        cross = DerivedVariable([self._xt, self._yt], cross_update)
        derived_base_variables += [cross]
        derived_base_variables_to_check = [cross]

        iters = 0
        while derived_base_variables_to_check:
            derived_base_variables_to_check, derived_base_variables = iterate_relations(derived_base_variables_to_check, self._base_variables, self._variable_dependence_graph, derived_base_variables)
            iters += 1
        return derived_base_variables, iters

    def derive_second_x(self):
        derived_base_variables = [DerivedVariable([self._xt, self._sin_thetat], None),
                                  DerivedVariable([self._xt, self._cos_thetat], None),
                                  DerivedVariable([self._yt, self._sin_thetat], None),
                                  DerivedVariable([self._yt, self._cos_thetat], None)]
        second_moment_update = sp.poly(self._xt.update_relation * self._xt.update_relation, self._base_variables.sympy_reps)
        second_moment = DerivedVariable([self._xt, self._xt], second_moment_update)
        derived_base_variables += [second_moment]
        derived_base_variables_to_check = [second_moment]
        self._xt._max_moment = 2
        print(second_moment.update_relation)
        iters = 0
        while derived_base_variables_to_check:
            derived_base_variables_to_check, derived_base_variables = iterate_relations(derived_base_variables_to_check, self._base_variables, self._variable_dependence_graph, derived_base_variables)
            iters += 1
        return derived_base_variables_to_check, iters

    def derive_fourth_x(self):
        derived_base_variables = [DerivedVariable([self._xt, self._sin_thetat], None),
                            DerivedVariable([self._xt, self._cos_thetat], None),
                            DerivedVariable([self._yt, self._sin_thetat], None),
                            DerivedVariable([self._yt, self._cos_thetat], None)]
        fourth_moment_update = sp.poly(self._xt.update_relation ** 4, self._base_variables.sympy_reps)
        fourth_moment = DerivedVariable([self._xt, self._xt, self._xt, self._xt], fourth_moment_update)
        derived_base_variables += [fourth_moment]
        derived_base_variables_to_check = [fourth_moment]
        for i in range(3):
            derived_base_variables_to_check, derived_base_variables = iterate_relations(derived_base_variables_to_check, self._base_variables, self._variable_dependence_graph, derived_base_variables)
            print(derived_base_variables_to_check)