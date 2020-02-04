from update_derivier import *
import sympy as sp
import networkx as nx

big_number = 10000

class TestThing(object):
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

        # Initialize base variables.
        self._xt = BaseVariable(xt, {1},  xt + vt * cos_thetat)
        self._yt = BaseVariable(yt, {1}, yt + vt * sin_thetat)
        self._vt = BaseVariable(vt, {1, 2}, vt + wvt)
        self._sin_thetat = BaseVariable(sin_thetat, set(range(big_number)), sp.expand_trig(sp.sin(thetat + wthetat)))
        self._cos_thetat = BaseVariable(cos_thetat, set(range(big_number)), sp.expand_trig(sp.cos(thetat + wthetat)))
        self._sin_wthetat = BaseVariable(sin_wthetat, set(range(big_number)), None)
        self._cos_wthetat = BaseVariable(cos_wthetat, set(range(big_number)), None)
        self._wvt = BaseVariable(wvt, set(range(big_number)), None)

        self._base_variables = [self._xt, self._yt, self._vt, self._sin_thetat, self._cos_thetat, self._sin_wthetat, self._cos_wthetat, self._wvt]

        self._variable_dependence_graph = nx.Graph()
        self._variable_dependence_graph.add_nodes_from(self._base_variables)
        self._variable_dependence_graph.add_edges_from([(self._xt, self._yt), (self._xt, self._vt), (self._yt, self._vt),
                                                        (self._xt, self._sin_thetat), (self._xt, self._cos_thetat), (self._yt, self._sin_thetat),
                                                        (self._yt, self._cos_thetat)])

        # These were derived in the ICRA paper.
        self._derived_variables = set()


    def print_derived_relations(self):
        for var in self._derived_variables:
            print("Variable " + str(var.sympy_rep))
            if var.update_relation:
                print(sp.expand(var.update_relation))
            print("\n")

    def test1(self):
        test_update = sp.poly(self._xt.update_relation * self._yt.update_relation, [var.sympy_rep for var in self._base_variables])
        res = test_iterate(test_update, self._base_variables, self._variable_dependence_graph, self._derived_variables)


    def test_max_order_2(self):
        integer_pairs = [(2, 0), (1,1), (0, 2)]
        for i, j in integer_pairs:
            test_update = sp.poly((self._xt.update_relation**i) * (self._yt.update_relation**j), [var.sympy_rep for var in self._base_variables])
            test_iterate(test_update, self._base_variables, self._variable_dependence_graph, self._derived_variables)
        self.print_derived_relations()

    def test_max_order_4(self):
        integer_pairs = [(2, 0), (1,1), (0, 2), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
        for i, j in integer_pairs:
            test_update = sp.poly((self._xt.update_relation**i) * (self._yt.update_relation**j), [var.sympy_rep for var in self._base_variables])
            test_iterate(test_update, self._base_variables, self._variable_dependence_graph, self._derived_variables)
        self.print_derived_relations()

    def test_eq_hash(self):
        foo1 = DerivedVariable({self._xt : 1, self._sin_thetat : 1}, None)
        foo2 = DerivedVariable({self._sin_thetat : 1, self._xt : 1}, None)
        assert foo1 == foo2
        set1 = {foo1}
        assert (foo2 in set1)
        set1.add(foo2)
        assert len(set1) == 1

tt = TestThing()
tt.setup()
tt.test_max_order_4()