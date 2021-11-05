%% Problem parameters
p = 5000;
n = 3000;
s = 40;
sigma = 0.05;

%% Generate data
nonzeros = randsample(p, s);
betastar = zeros(p, 1);
betastar(nonzeros) = randn(s, 1);
betastar = betastar/vecnorm(betastar); % Unit-normalize

X = randn(n, p);
Y = (X*betastar).^2 + sigma*randn(n, 1);

%% Matrix optimization
lambda = 4*(sigma+1e-4)*sqrt(s*(1 + log(p/s)) / n);
[Uhat, Vhat, fval] = pr_opt_altmin(X, Y, lambda, s);

%% Get vector estimate
[V_Usvd, V_Ssvd, ~] = svd(Vhat, 'econ');
betahat = sqrt(V_Ssvd(1,1)*norm(Uhat))*V_Usvd(:, 1);
signerr = sign(betahat'*betastar);
error = norm(signerr*betahat - betastar)