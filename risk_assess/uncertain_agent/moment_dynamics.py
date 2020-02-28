from risk_assess.random_objects.trigonometric_moments import CosSumOfRVs, SinSumOfRVs, CrossSumOfRVs
from copy import deepcopy
import time

def propagate_moments(initial_state, w_vs, w_thetas):
    """
    Given some initial 
    Args:
        state (instance of AgentMomentState)
        w_vs (list of instances of RandomVariable)
        w_thetas (list of instances of RandomVariable)
    """
    states = [initial_state]
    assert len(w_thetas) == len(w_vs)
    for i in range(len(w_thetas)):
        new_state = propagate_one_step(states[i], w_vs[i], w_thetas[i])
        states.append(new_state)
    return states

def propagate_one_step(state, w_v, w_theta):
    c = state.theta.cos_applied()
    s = state.theta.sin_applied()
    cs = CrossSumOfRVs(state.theta.c, state.theta.random_variables)
    E_c = c.compute_moment(1)
    E_s = s.compute_moment(1)
    E2_c = c.compute_moment(2)
    E2_s = s.compute_moment(2)
    E_cs = cs.compute_moment(1)

    # Compute moments of control variables.
    E_cw = CosSumOfRVs(0, [w_theta]).compute_moment(1)
    E_sw = SinSumOfRVs(0, [w_theta]).compute_moment(1)
    E_wv = w_v.compute_moment(1)
    E2_wv = w_v.compute_moment(2)
    
    # Renaming old moments.
    E_v = state.E_v
    E2_v = state.E2_v
    E_xs = state.E_xs
    E_ys = state.E_ys
    E_xc = state.E_xc
    E_yc = state.E_yc
    E_xvs = state.E_xvs
    E_xvc = state.E_xvc
    E_yvs = state.E_yvs
    E_yvc = state.E_yvc
    E_x = state.E_x
    E_y = state.E_y
    E_xy = state.E_xy
    E2_x = state.E2_x
    E2_y = state.E2_y

    #
    new_state = deepcopy(state)

    new_state.theta.add_rv(w_theta)

    new_state.E_v = E_v + E_wv

    new_state.E2_v = E2_v + 2 * E_v * E_wv + E2_wv

    new_state.E_xs = E_v*E_cs*E_cw + E_v*E_sw*E2_c + E_xs*E_cw + E_xc * E_sw

    new_state.E_ys = E_v*E2_s*E_cw + E_v*E_sw*E_cs + E_ys*E_cw + E_yc * E_sw

    new_state.E_xc = -E_v*E_cs * E_sw + E_v*E2_c*E_cw - E_xs*E_sw + E_xc*E_cw

    new_state.E_yc = -E_v*E2_s*E_sw + E_v*E_cs*E_cw - E_ys*E_sw + E_yc*E_cw

    new_state.E_xvs = E2_v*E_cs*E_cw + E2_v*E_sw*E2_c+ E_v*E_wv*E_cs*E_cw + E_v*E_wv*E_sw*E2_c + E_xvs*E_cw +\
          E_xvc * E_sw + E_wv * E_xs * E_cw  + E_wv * E_xc * E_sw

    new_state.E_xvc = -E2_v*E_cs*E_sw + E2_v*E2_c*E_cw- E_v*E_wv*E_cs*E_sw+ E_v*E_wv*E2_c*E_cw -\
          E_xvs*E_sw + E_xvc*E_cw- E_wv*E_xs*E_sw + E_wv*E_xc*E_cw

    new_state.E_yvs = E2_v*E2_s*E_cw+ E2_v*E_cs*E_sw + E_v*E_wv*E2_s*E_cw+ E_v*E_wv*E_cs*E_sw +\
          E_yvs*E_cw+ E_yvc*E_sw + E_wv*E_ys*E_cw+ E_wv*E_yc*E_sw

    new_state.E_yvc = -E2_v*E2_s*E_sw + E2_v*E_cs*E_cw- E_v*E_wv*E2_s*E_sw + E_v*E_wv*E_cs*E_cw-\
          E_yvs*E_sw + E_yvc*E_cw- E_wv*E_ys*E_sw + E_wv*E_yc*E_cw
    
    new_state.E_x = E_x + E_v * E_c

    new_state.E_y = E_y + E_v * E_s

    new_state.E_xy = E2_v * E_cs + E_xvs + E_yvc + E_xy

    new_state.E2_x = E2_v*E2_c + 2*E_xvc + E2_x

    new_state.E2_y = E2_v*E2_s + 2*E_yvs + E2_y
    return new_state