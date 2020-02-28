import math
from risk_assess.random_objects.trigonometric_moments import SumOfRVs
from dataclasses import dataclass

@dataclass
class AgentMomentState:
    E_x: float
    E_y: float
    E_xy: float
    E2_x: float
    E2_y: float
    E_xvs: float
    E_xvc: float
    E_yvs: float
    E_yvc: float
    E_xs: float
    E_xc: float
    E_ys: float
    E_yc: float
    E_v: float
    E2_v: float
    theta: SumOfRVs

    @classmethod
    def from_deterministic_state(cls, x0, y0, v0, theta0, numerical_padding = 0.0):
        return cls(
            E_x = x0,
            E_y = y0,
            E_xy = x0 * y0,
            E2_x = x0**2 + numerical_padding,
            E2_y = y0**2 + numerical_padding,
            E_xvs = x0 * v0 * math.sin(theta0),
            E_xvc = x0 * v0 * math.cos(theta0),
            E_yvs = y0 * v0 * math.sin(theta0),
            E_yvc = y0 * v0 * math.cos(theta0),
            E_xs = x0 * math.sin(theta0),
            E_xc = x0 * math.cos(theta0),
            E_ys = y0 * math.sin(theta0),
            E_yc = y0 * math.cos(theta0),
            E_v = v0,
            E2_v = v0**2 + numerical_padding,
            theta = SumOfRVs(theta0, []))
    
    def speed_scaled(self, scaling_factor):
        return AgentMomentState.from_deterministic_state(self.E_x, self.E_y, scaling_factor * self.E_v, self.theta.c)