function [beta_moments] = calc_beta_moments(alpha,beta,k)
%Generate the moments of a beta distribution with parameters alpha and beta
%up to moment number k
beta_moments = [alpha/(alpha+beta)]; %first moment
for i = 2:k
    r = i-1;
    beta_moments(i) = beta_moments(i-1)*((alpha+r)/(alpha+beta+r));
end
end

