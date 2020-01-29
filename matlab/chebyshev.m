addpath("../Time Invariant/ForwardsSets/")
addpath("../utils/")

f = @(x,y) -2*x.^2 -4*y.^2 + 3*y.^4 + 2.5*x.^4-0.1;
fimplicit(f)

%Params for beta distribution
alpha = 500;
beta = 550;
X = 0:0.01:1;
y = betapdf(X,alpha,beta);
figure()
plot(2.*X,y)
title(['2*Beta(',num2str(alpha),',',num2str(beta),')'])

mu0 = [2;2]; %initial mean values for [x,y]
sigma0 = [0.5,0.1;0.1,0.3]; %initial covariance matrix
sigmaw = [0.03,0.01;0.01,0.03]; %noise injected at each time step

A = [1.0,0.01;-0.04,1.0];
B = [1,1];
C = [0.1,0.02;0.05,0.04];
n = 10;
mus = [mu0];

for i = 2:n
    mus(:,i) = mus(:,i-1) + [-0.1;-0.1];
    sigmaxk = sigma0;
    for j = 1:i-1
       sigmaxk = sigmaxk + (A^j)*C*sigmaw*C'*(A')^j;
    end
end
