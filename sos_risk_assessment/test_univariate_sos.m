r = normrnd(5, 1, 1000, 1);

mus = [1, 1, 1.5, 2.0, 3.0];
rb = univariate_sos_risk_bound(mus);