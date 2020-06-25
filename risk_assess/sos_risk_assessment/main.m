% Load moments computed in Python.
load order6.mat

% Specify maximum order of moments to use.
max_order = 6;
n_cases = size(collision_moments, 1);

% Result arrays.
risk_bounds = zeros(1, n_cases);
flags = zeros(1, n_cases);
solve_times = zeros(1, n_cases);

for i = 15:n_cases
    % Normalize the moments to improve numerical conditioning.
    normalized = normalize_moments(collision_moments(i, 1:max_order));
    
    % Solve the problem and store results.
    [rb, f, sol]  = univariate_sos_risk_bound(normalized, "sedumi");
    risk_bounds(i) = rb;
    flags(i) = f;
    solve_times(i) = sol.solvertime;
end

chebyshev_risk_bounds = zeros(1, n_cases);
for i = 1:n_cases
    if collision_moments(i, 1) <= 0
        chebyshev_risk_bounds(i) = nan;
    else
        chebyshev_risk_bounds(i) = (collision_moments(i, 2) - collision_moments(i, 1)^2)/collision_moments(i,2);
    end
end
