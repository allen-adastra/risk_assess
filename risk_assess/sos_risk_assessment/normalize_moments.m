function [moments] = normalize_moments(moments)
%Normalize an array of moments s.t. the maximum is 1.
%   Input moments of order [first, second, third, ...]
max_order = length(moments);

% Compute the scaling constant "c" s.t. the nth order moment gets
% scaled by c^n. We compute c by letting c^n = 1.0/moments(end)
c = (1.0/moments(end))^(1.0/max_order);

for order = 1:max_order
   moments(order) = (c^order)*moments(order); 
end
end