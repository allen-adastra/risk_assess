% load data/position_gmm_component_scenario_7_component_0.mat
load data/position_gmm_component_scenario_13_component_0.mat

% The moments were computed for the event Q <=1, need to recompute
% For the event Q^* <= 0
for i = 1:30
   for j = 2:6
      moments(i, j) = (moments(i, j) - 1)^(j - 1);
   end
end

dim_moments = size(moments);
n_step = dim_moments(1);
risk_bounds = zeros(1, n_step);
flags = zeros(1, n_step);
solve_times = zeros(1, n_step);
parfor i = 1:n_step
    [rb, f, sol]  = univariate_sos_risk_bound(moments(i, 1:5));
    risk_bounds(i) = rb;
    flags(i) = f;
    solve_times(i) = sol.solvertime;
end

chebyshev_risk_bounds = zeros(1, n_step);
for i = 1:n_step
    if moments(i, 2) <= 0
        chebyshev_risk_bounds(i) = nan;
    else
        chebyshev_risk_bounds(i) = (moments(i, 3) - moments(i, 2)^2)/moments(i,3);
    end
end