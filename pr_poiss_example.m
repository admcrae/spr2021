%% Problem parameters
p = 5000;
n = 3000;
s = 40;
beta_norm = 20;

%% Generate data
nonzeros = randsample(p, s);
betastar = zeros(p, 1);
betastar(nonzeros) = randn(s, 1);
betastar = beta_norm*betastar/vecnorm(betastar);

X = randn(n, p);
Y = poissrnd((X*betastar).^2);

%% Matrix optimization
lambda = 3*(1+beta_norm)*sqrt(s*(1 + log(p/s)) / n);
[Uhat, Vhat, fval] = pr_opt_altmin(X, Y, lambda, s);

%% Get vector estimate
[V_Usvd, V_Ssvd, ~] = svd(Vhat, 'econ');
betahat = sqrt(V_Ssvd(1,1)*norm(Uhat))*V_Usvd(:, 1);
error = phaseless_err(betahat, betastar)