from automated_update_derivations import identify_needed_updates, BaseVariable, BaseVariables
import sympy as sp
import networkx as nx

class TestClass:
    def setup(self):
        # Declare variables at time "t"
        self._xt = sp.Symbol("x_t")
        self._yt = sp.Symbol("y_t")
        self._vt = sp.Symbol("v_t")
        self._wvt = sp.Symbol("w_{v_t}")
        self._wthetat = sp.Symbol("w_{theta_t}")
        self._thetat = sp.Symbol("theta_t")
        self._cos_thetat = sp.cos(self._thetat)
        self._sin_thetat = sp.sin(self._thetat)
        self._sin_thetat = sp.sin(self._thetat)
        self._cos_thetat = sp.cos(self._thetat)
        self._sin_wthetat = sp.sin(self._wthetat)
        self._cos_wthetat = sp.cos(self._wthetat)

        self._xt_base = BaseVariable(self._xt, 1,  self._xt + self._vt * self._cos_thetat)
        self._yt_base = BaseVariable(self._yt, 1, self._yt + self._vt * self._sin_thetat)
        self._vt_base = BaseVariable(self._vt, 2, self._vt + self._wvt)
        self._sin_thetat_base = BaseVariable(self._sin_thetat, 1000, sp.expand_trig(sp.sin(self._thetat + self._wthetat)))
        self._cos_thetat_base = BaseVariable(self._cos_thetat, 1000, sp.expand_trig(sp.cos(self._thetat + self._wthetat)))
        self._sin_wthetat_base = BaseVariable(self._sin_wthetat, 0, None)
        self._cos_wthetat_base = BaseVariable(self._cos_wthetat, 0, None)

        self._base_variables = BaseVariables([self._xt_base, self._yt_base, self._vt_base, self._sin_thetat_base,
                                            self._cos_thetat_base, self._sin_wthetat_base, self._cos_wthetat_base])

        self._variable_dependence_graph = nx.Graph()
        self._variable_dependence_graph.add_nodes_from(self._base_variables.sympy_rep)
        self._variable_dependence_graph.add_edges_from([(self._xt, self._yt), (self._xt, self._vt), (self._yt, self._vt),
                                                        (self._xt, self._sin_thetat), (self._xt, self._cos_thetat), (self._yt, self._sin_thetat),
                                                        (self._yt, self._cos_thetat)])


    def test_cross(self):
        self.setup()
        terms_with_generated_relations = [set([self._xt, self._sin_thetat]), set([self._xt, self._cos_thetat]),
                                          set([self._yt, self._sin_thetat]), set([self._yt, self._cos_thetat])]

        # Construct the expression for E[x_{t+1} * y_{t+1}] in terms of expressions at time t
        cross = sp.poly(self._xt_base.update_relation * self._yt_base.update_relation, self._base_variables.sympy_rep)
        terms_with_generated_relations += [set([self._xt, self._yt])]
        new_update_relations_needed, variables_need_higher_moments = identify_needed_updates(cross, self._base_variables, self._variable_dependence_graph, terms_with_generated_relations)

        assert len(new_update_relations_needed) == 2
        assert ({self._xt, self._vt, self._sin_thetat} in new_update_relations_needed)
        assert ({self._yt, self._vt, self._cos_thetat} in new_update_relations_needed)
        assert len(variables_need_higher_moments) == 0