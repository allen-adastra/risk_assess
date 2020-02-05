import pytest
import tree_ring.objects as tro
from tree_ring.systems.uncontrolled_agent import UncontrolledAgent

def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()

    
def test_eq_hash():
    """
    Test that if two instances of DerivedVariable are instantiated with the same
    """
    unc_ag = UncontrolledAgent()
    foo1 = tro.DerivedVariable({unc_ag._xt : 1, unc_ag._sin_thetat : 1}, None)
    foo2 = tro.DerivedVariable({unc_ag._sin_thetat : 1, unc_ag._xt : 1}, None)
    assert foo1 == foo2
    set1 = {foo1}
    assert (foo2 in set1)
    set1.add(foo2)
    assert len(set1) == 1