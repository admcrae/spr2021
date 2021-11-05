function ymin = prox_l1(g, x, alpha, beta)
% Solves argmin_y g'*y + 0.5*||x - y||_2^2 + 0.5*alpha*||y||_2^2 + 0.5*beta*||y||_1^2


% Convert to
%   argmax_y (x - g)'*y - 0.5*(1+alpha)*||y||_2^2 - 0.5*beta*||y||_1^2
% = argmax_y ((x-g)/(1+alpha)) - 0.5*||y||_2^2 
%       - 0.5*(beta/(1 +alpha))*||y||_1^2

[n, ydual] = dualsnorm((x - g)/(1 + alpha), (1+alpha)/beta);
ymin = n.*ydual./thetasnorm(ydual, (1 + alpha)/beta);

end