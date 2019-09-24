"""
Discrete time kinematic car model that stores symbolic expressions for the
cars state at time steps up to n_steps.
"""
class KinematicCarAccelNoise(object):
    def __init__(self, n_steps, dt):
        x0 = sp.symbols('x0', real = True, constant = True)
        y0 = sp.symbols('y0', real = True, constant = True)
        v0 = sp.symbols('v0', real = True, constant = True)

        # uaw is the accel command multiplied by a random variable for accel noise.
        accel_mult_random = sp.symbols('uaw0:' + str(n_steps), real = True)

        # vs is a list of speeds at times 0, 1,..., n_steps
        dvs = list(accumulate([0] + list(accel_mult_random), lambda cum_sum, next_accel: cum_sum + next_accel))
        vs = [v0 + dt*dv for dv in dvs]

        thetas = sp.symbols('theta0:' + str(n_steps), real = True)
        cos_thetas = [sp.cos(t) for t in thetas]
        sin_thetas = [sp.sin(t) for t in thetas]

        xs = list(accumulate([x0] + [x for x in range(n_steps)], lambda pre,i: pre + vs[i]*cos_thetas[i]))
        ys = list(accumulate([y0] + [x for x in range(n_steps)], lambda pre,i: pre + vs[i]*sin_thetas[i]))

        # Get xs and ys in polynomial form.
        self.xs = [sp.poly(x, accel_mult_random) for x in xs]
        self.ys = [sp.poly(y, accel_mult_random) for y in ys]
        self.n_steps = n_steps
        self.dt = dt
        self.x0 = x0
        self.y0 = y0
        self.v0 = v0
        self.thetas = thetas

    def get_final_state_vec(self):
        return np.asarray([self.xs[-1], self.ys[-1]])
