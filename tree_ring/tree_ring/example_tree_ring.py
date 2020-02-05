from tree_ring.tree_ring import tree_ring
from tree_ring.systems.uncontrolled_agent import UncontrolledAgent
import sympy as sp
import networkx as nx

class TestThing(UncontrolledAgent):
    def print_derived_relations(self):
        for var in self._derived_variables:
            print("Variable " + str(var.sympy_rep))
            if var.update_relation:
                print(sp.expand(var.update_relation))
            print("\n")

    def test1(self):
        test_update = sp.poly(self._xt.update_relation * self._yt.update_relation, [var.sympy_rep for var in self._base_variables])
        res = tree_ring(test_update, self._base_variables, self._variable_dependence_graph, self._derived_variables)


    def test_max_order_2(self):
        integer_pairs = [(2, 0), (1,1), (0, 2)]
        for i, j in integer_pairs:
            var_power_map = {self._xt : i, self._yt : j}
            tree_ring(var_power_map, self._base_variables, self._variable_dependence_graph, self._derived_variables)
        self.print_derived_relations()

    def test_max_order_4(self):
        integer_pairs = [(2, 0), (1,1), (0, 2), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
        for i, j in integer_pairs:
            var_power_map = {self._xt : i, self._yt : j}
            tree_ring(var_power_map, self._base_variables, self._variable_dependence_graph, self._derived_variables)
        self.print_derived_relations()



tt = TestThing()
tt.test_max_order_2()