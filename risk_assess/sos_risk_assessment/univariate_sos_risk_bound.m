function [risk_bound, flag, sol] = univariate_sos_risk_bound(mus, solver)
% For some random variable X, upper bound the probability:
% Prob(X <= 0)
% By solving a univariate SOS program.
% mus: vector of moments of X [X^1, X^2, ...]

% Zeroth moment is 1.
mus = [1, mus];

n = numel(mus); % Number of moments
max_order = n - 1; % Maximum order of moments and the p polynomial
d = floor(max_order/2); % order = 2d or 2d + 1

% Generate the indeterminate variable x and its powers
x = sdpvar(1);
xs = monolist(x, max_order);
cs = sdpvar(1, n); % Coefficients for the polynomial p.
p = cs * xs; % The polynomial p that will upper bound the indicator function.

% "Multiplier" SOS Polynomials s1 and s2 to search for
if mod(max_order, 2) == 0
    s1_coeff = sdpvar(1, 2 * d + 1); % Need 2d + 1 coefficients
    s1_basis = monolist(x, 2 * d); % Up to 2d degree
    s1 = s1_coeff * s1_basis;
    s2_coeff = sdpvar(1, 2 * d - 1);
    s2_basis = monolist(x, 2 * d - 2);
    s2 = s2_coeff * s2_basis;
else
    s1_coeff = sdpvar(1, 2 * d + 1);
    s1_basis = monolist(x, 2 * d);
    s1 = s1_coeff * s1_basis;
    s2_coeff = sdpvar(1, 2 * d + 1);
    s2_basis = monolist(x, 2 * d);
    s2 = s2_coeff * s2_basis;
end

% We want the constraint p - 1 = s1 - x * s2
% 
lhs = p - 1;
rhs = s1 - x * s2;
coeffs_rhs = coefficients(rhs, x);
coeffs_lhs = coefficients(lhs , x);
F = [coeffs_lhs == coeffs_rhs; sos(p); sos(s1); sos(s2)];
obj = cs * mus';
ops = sdpsettings('solver', solver);
sol = solvesos(F,obj,ops,[cs, s1_coeff, s2_coeff]);
risk_bound = value(obj);
flag = sol.problem;
if flag ~=0
   risk_bound = nan; 
end
end