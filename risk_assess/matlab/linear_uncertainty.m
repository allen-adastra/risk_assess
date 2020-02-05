addpath("../Time Invariant/ForwardsSets/")
addpath("../utils/")

n = 5;
dt = 0.05;
w = sym('w',n);


%moments, assume everything is distributed according to the following beta
%distribution
alpha = 500;
beta = 550;
n_moments = 200;
beta_moments = calc_beta_moments(alpha,beta, n_moments);


%we want two times the beta distribution as our random variable
for i = 1:n_moments
   beta_moments(i) = (2^i)*beta_moments(i);
end



