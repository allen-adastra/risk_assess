addpath("../Time Invariant/ForwardsSets/")
addpath("../utils/")

n = 5;
dt = 0.05;
taylor_order = 3;

w = sym('w',n);
syms wtheta0 wx0 wy0

%moments, assume everything is distributed according to beta
alpha = 500;
beta = 550;
n_moments = 200;
beta_moments = calc_beta_moments(alpha,beta, n_moments);

%we want two times the beta distribution as our random variable
for i = 1:n_moments
   beta_moments(i) = (2^i)*beta_moments(i);
end


%Control inputs
vs = ones(1,n);
steers = 0.01*ones(1,n);


%Initial state with uncertainty
x0 = 1.0 + wx0;
y0 = 0 + wy0;
theta0 = pi + wtheta0;

%state vectors across time
xs = [x0];
ys = [y0];
thetas = [theta0];


for i = 2:n
    thetas(i) = thetas(i-1) - 0.1;
    xs(i) = xs(i-1) + dt*vs(i)*cos_taylor(thetas(i), taylor_order)*w(i,1);
    ys(i) = ys(i-1) + dt*vs(i)*sin_taylor(thetas(i), taylor_order);
end
xs = subs(xs, [wx0, wy0, wtheta0], [0,0,0]);
ys = subs(ys, [wx0, wy0, wtheta0], [0,0,0]);

expected_xs = subs(xs, [w(:,1)], [repmat(beta_moments(1), size(w,1),1)]);
expected_ys = subs(ys, [w(:,1)], [repmat(beta_moments(1), size(w,1),1)]);


thetas = subs(thetas, [wx0, wy0, wtheta0], [0,0,0]);

p = @(x,y) -2*x.^2 -4*y.^2 + 3*y.^4 + 2.5*x.^4-0.1;
fimplicit(p)
final_state_p = expand(p(xs(n),ys(n)));
final_state_p_sq = expand(final_state_p*final_state_p);

final_x = xs(n);
final_y = ys(n);
final_x_sq = expand(final_x*final_x);
final_y_sq = expand(final_y*final_y);


for i = 2:n
   for j = n_moments:-1:1
      %need to do this shit backwards
      final_state_p = subs(final_state_p,[w(i,1)^j],[beta_moments(j)]);
      final_state_p_sq = subs(final_state_p_sq,[w(i,1)^j],[beta_moments(j)]);
      final_x = subs(final_x, [w(i,1)^j], [beta_moments(j)]);
      final_y = subs(final_y, [w(i,1)^j], [beta_moments(j)]);
      final_x_sq = subs(final_x_sq, [w(i,1)^j], [beta_moments(j)]);
      final_y_sq = subs(final_y_sq, [w(i,1)^j], [beta_moments(j)]);
   end
end

first_moment = double(final_state_p);
second_moment = double(final_state_p_sq);
variance = second_moment - first_moment^2;




bound_prob_fail = chebyshev_prob_fail_bound(first_moment, second_moment)

