from automated_update_derivations import identify_needed_updates, BaseVariable, BaseVariables, DerivedVariable, iterate_relations
import sympy as sp
import networkx as nx

class TestClass:
    def setup(self):
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

        self._xt = BaseVariable(xt, 1,  xt+ vt * cos_thetat)
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


    def test_cross(self):
        self.setup()
        # These were derived in the ICRA paper.
        derived_base_variables = [DerivedVariable([self._xt, self._sin_thetat], None),
                                  DerivedVariable([self._xt, self._cos_thetat], None),
                                  DerivedVariable([self._yt, self._sin_thetat], None),
                                  DerivedVariable([self._yt, self._cos_thetat], None)]
        # Construct the expression for E[x_{t+1} * y_{t+1}] in terms of expressions at time t
        cross_update = sp.poly(self._xt.update_relation * self._yt.update_relation, self._base_variables.sympy_reps)
        cross = DerivedVariable([self._xt, self._yt], cross_update)
        derived_base_variables += [cross]
        new_update_relations_needed, variables_need_higher_moments = identify_needed_updates(cross, self._base_variables, self._variable_dependence_graph, derived_base_variables)
        assert len(new_update_relations_needed) == 2
        assert ({self._xt, self._vt, self._sin_thetat} in new_update_relations_needed)
        assert ({self._yt, self._vt, self._cos_thetat} in new_update_relations_needed)
        assert len(variables_need_higher_moments) == 0

    def test_iterate_relations(self):
        self.setup()

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
        # It should take three iterations for the given problem.
        assert iters == 3 
        # The resulting derived variables should be the following nine, and no others.
        derived_base_variables_components = [set(var.component_variables) for var in derived_base_variables]
        assert {self._xt, self._sin_thetat} in derived_base_variables_components
        assert {self._xt, self._cos_thetat} in derived_base_variables_components
        assert {self._yt, self._sin_thetat} in derived_base_variables_components
        assert {self._yt, self._cos_thetat} in derived_base_variables_components
        assert {self._xt, self._yt} in derived_base_variables_components
        assert {self._xt, self._vt, self._sin_thetat} in derived_base_variables_components
        assert {self._xt, self._vt, self._cos_thetat} in derived_base_variables_components
        assert {self._yt, self._vt, self._sin_thetat} in derived_base_variables_components
        assert {self._yt, self._vt, self._cos_thetat} in derived_base_variables_components
        assert len(derived_base_variables_components) == 9
        for var in derived_base_variables:
            print("Update relation for " + str(var.component_variables_sympy))
            print(var.update_relation)
            
    def test_derived_base_var(self):
        x = BaseVariable(sp.Symbol('x'), 1, None)
        y = BaseVariable(sp.Symbol('y'), 1, None)
        x_times_y = DerivedVariable([x, y], None)
        assert x_times_y.string_rep == 'x * y' # Syntax for the string representation of instances of DerivedVariable
        assert x_times_y.component_variables == [x, y]
        assert x_times_y.sympy_equivalent({x.sympy_rep, y.sympy_rep}) == True
        assert x_times_y.sympy_equivalent({y.sympy_rep, x.sympy_rep}) == True
        assert x_times_y.sympy_equivalent({x.sympy_rep, y.sympy_rep, sp.Symbol('z')}) == False