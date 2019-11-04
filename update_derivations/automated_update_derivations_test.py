from automated_update_derivations import identify_needed_updates, BaseVariable, BaseVariables
import sympy as sp
import networkx as nx

class TestClass:
    def test_cross(self):
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

        terms_with_generated_relations = [set([xt, sin_thetat]), set([xt, cos_thetat]), set([yt, sin_thetat]), set([yt, cos_thetat])]

        # Construct the expression for E[x_{t+1} * y_{t+1}] in terms of expressions at time t
        cross = sp.poly(xt_base.update_relation * yt_base.update_relation, base_variables.sympy_rep)
        terms_with_generated_relations += [set([xt, yt])]
        new_update_relations_needed, variables_need_higher_moments = identify_needed_updates(cross, base_variables, variable_dependence_graph, terms_with_generated_relations)

        assert len(new_update_relations_needed) == 2
        assert ({xt, vt, sin_thetat} in new_update_relations_needed)
        assert ({yt, vt, cos_thetat} in new_update_relations_needed)
        assert len(variables_need_higher_moments) == 0