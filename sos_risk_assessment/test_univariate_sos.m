load data/position_gmm_component_scenario_7_component_0.mat
dim_moments = size(moments);
n_step = dim_moments(1);
risk_bounds = zeros(1, n_step);
flags = zeros(1, n_step);
for i = 1:n_step
    [rb, f]  = univariate_sos_risk_bound(moments(i, 1:5));
    risk_bounds(i) = rb;
    flags(i) = f;
end

chebyshev_risk_bounds = zeros(1, n_step);
for i = 1:n_step
    chebyshev_risk_bounds(i) = (moments(i, 3) - moments(i, 2)^2)/moments(i,3);
end