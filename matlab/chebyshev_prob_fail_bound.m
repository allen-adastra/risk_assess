function [chebyshev_bound] = chebyshev_prob_fail_bound(first_moment, second_moment)
%Given some semialgebraic set with failure defined as p<=0, 
%first_moment is the first moment of p and second_moment is the second
%moment of p, then this function returns a bound on the probability of
%failure via Chebyshev's inequality
first_moment_sq = first_moment^2;
variance = second_moment - first_moment_sq;

if first_moment>0
   %Then we can apply P(p<= E[p] - a) <= (sigma^2)/(sigma^2 + a^2)
   %set a = E[p] and then P(p <= 0) <= (sigma^2)/(sigma^2 + E[p]^2)
   chebyshev_bound = variance/(variance + first_moment_sq);
else
    
    %In this case, we will look for an upper bound on the probability that
    %it does NOT fail
    
    chebyshev_bound = variance/first_moment_sq;
    
    
    %We need to find:
    %P(|p - E[p]| >= t) + P(p <= E[p] - t)
    
    %Apply the two sided inequality to bound the first term
    %P(|p - E[p]| >= t) <= variance/t^2
    %P(|p - E[p]| >= E[p]) <= variance/E[p]^2
    
    %Apply the one sided inequality to bound the second term
    %P(p <= E[p] - |E[p]|) <= variance/(variance + E[p]^2)
    
    %So really the whole bound is just
    
    
    
    
    
end

end

