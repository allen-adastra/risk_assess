from risk_assess.tree_ring import *
import risk_assess.tree_ring.objects as tro
import sympy as sp
import networkx as nx

big_number = 10000
class UncontrolledAgent(object):
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

        # Initialize base variables.
        self._xt = tro.BaseVariable(xt, {1},  xt + vt * cos_thetat)
        self._yt = tro.BaseVariable(yt, {1}, yt + vt * sin_thetat)
        self._vt = tro.BaseVariable(vt, {1, 2}, vt + wvt)
        self._sin_thetat = tro.BaseVariable(sin_thetat, set(range(big_number)), sp.expand_trig(sp.sin(thetat + wthetat)))
        self._cos_thetat = tro.BaseVariable(cos_thetat, set(range(big_number)), sp.expand_trig(sp.cos(thetat + wthetat)))
        self._sin_wthetat = tro.BaseVariable(sin_wthetat, set(range(big_number)), None)
        self._cos_wthetat = tro.BaseVariable(cos_wthetat, set(range(big_number)), None)
        self._wvt = tro.BaseVariable(wvt, set(range(big_number)), None)

        self._base_variables = [self._xt, self._yt, self._vt, self._sin_thetat, self._cos_thetat, self._sin_wthetat, self._cos_wthetat, self._wvt]

        self._variable_dependence_graph = nx.Graph()
        self._variable_dependence_graph.add_nodes_from(self._base_variables)
        self._variable_dependence_graph.add_edges_from([(self._xt, self._yt), (self._xt, self._vt), (self._yt, self._vt),
                                                        (self._xt, self._sin_thetat), (self._xt, self._cos_thetat), (self._yt, self._sin_thetat),
                                                        (self._yt, self._cos_thetat)])

        # These were derived in the ICRA paper.
        self._derived_variables = set()