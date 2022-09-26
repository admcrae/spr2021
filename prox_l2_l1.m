function ymin = prox_l2_l1(g, x, alpha, beta)
% Solves argmin_y g'*y + 0.5*||x - y||_2^2 + alpha||y||_2 + beta*||y||_1

% Same as argmin_y (g - x)'*y + 0.5*(C+1)*||y||_2^2 + beta*||y||_1 for some
% C > 0 that we search over
% The subproblem solution is simple soft thresholding. It is optimal for
% the original problem if C = alpha/||y||_2, where y is the subproblem
% solution. We can then explicitly solve for C
gnew = g - x;

y_unscaled = -sign(gnew).*max(abs(gnew) - beta, 0);
n = vecnorm(y_unscaled, 2, 1);

ymin = y_unscaled .* max(1 - alpha./n, 0);
end