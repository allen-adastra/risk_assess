from risk_assess.uncertain_agent.state_objects import AgentMomentState
from risk_assess.uncertain_agent.moment_dynamics import propagate_moments
from risk_assess.random_objects.random_variables import Normal
from risk_assess.deterministic import simulate_deterministic
import numpy as np

def relative_error(true_val, measured):
    return abs(measured/true_val - 1)

def test_propagated_moments():
    """
    Test the accuracy of the moments propagated with 'propagate_moments' against Monte Carlo.
    """
    # Test data.
    n_step = 30
    dt = 1.0
    n_samps = int(1e6)
    relative_error_tolerance = 1e-3
    x0 = 1
    y0 = 1
    v0 = 1
    theta0 = 1
    initial_state = AgentMomentState.from_deterministic_state(x0, y0, v0, theta0)
    wvs = [Normal(0.1, 0.01) for i in range(n_step)]
    wthetas = [Normal(0.1, 0.01) for i in range(n_step)]

    # Simulate the system with Monte Carlo
    accel_samps = np.hstack([rv.sample(n_samps) for rv in wvs])
    steer_samps = np.hstack([rv.sample(n_samps) for rv in wthetas])
    xs, ys, _, _ = simulate_deterministic(x0, y0, v0, theta0, accel_samps, steer_samps, dt)
    
    # Propagate moments, make sure the number of time steps is consistent between the two methods.
    moments = propagate_moments(initial_state, wvs, wthetas)
    assert len(moments) == xs.shape[1]
    
    # Final state quantities.
    final_state_moment = moments[-1]
    final_xs = xs[:, -1]
    final_ys = ys[:, -1]

    # Compute relative errors and check its below the tolerance.
    expected_x_err = relative_error(np.mean(final_xs), final_state_moment.E_x)
    expected_y_err = relative_error(np.mean(final_ys), final_state_moment.E_y)
    expected_xy_err  = relative_error(np.mean(np.multiply(final_xs, final_ys)), final_state_moment.E_xy)
    second_moment_x_err = relative_error(np.mean(np.power(final_xs, 2)), final_state_moment.E2_x)
    second_moment_y_err = relative_error(np.mean(np.power(final_ys, 2)), final_state_moment.E2_y)
    assert max([expected_x_err, expected_y_err, expected_xy_err, second_moment_x_err, second_moment_y_err]) < relative_error_tolerance