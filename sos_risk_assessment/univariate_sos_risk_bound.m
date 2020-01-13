function [] = univariate_sos_risk_bound(mus)
% mus: vector of moments [zeroth order, first order, second order, ...]

n = numel(mus); % Number of moments
max_order = n - 1; % Maximum order of moments and the p polynomial
d = floor(max_order/2); 

% Generate the indeterminate variable x and its powers
x = sdpvar(1);
xs = monolist(x,max_order);
cs = sdpvar(1, n);
p = cs * xs;


if mod(max_order, 2) == 0
    % SOS Polynomials s1 and s2 to search for
    s1_coeff = sdpvar(1, 2 * d + 1);
    s1_basis = monolist(x, 2 * d);
    s1 = s1_coeff * s1_basis;
    s2_coeff = sdpvar(1, 2 * d - 1);
    s2_basis = monolist(x, 2 * d - 2);
    s2 = s2_coeff * s2_basis;
else
    % SOS Polynomials s1 and s2 to search for
    s1_coeff = sdpvar(1, 2 * d + 1);
    s1_basis = monolist(x, 2 * d);
    s1 = s1_coeff * s1_basis;
    s2_coeff = sdpvar(1, 2 * d + 1);
    s2_basis = monolist(x, 2 * d);
    s2 = s2_coeff * s2_basis;
end
% We want the constraint p - 1 = s1 - x * s2
lhs = p - 1;
rhs = s1 - x * s2;
coeffs_rhs = coefficients(rhs, x);
coeffs_lhs = coefficients(lhs , x);
F = [coeffs_lhs == coeffs_rhs; sos(p); sos(s1); sos(s2)];
obj = cs * mus';
ops = sdpsettings('solver', 'scs');
sol = solvesos(F,obj,ops,[cs, s1_coeff, s2_coeff]);
end

