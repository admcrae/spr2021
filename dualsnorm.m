function [n, u] = dualsnorm(v, s)
% Returns the maximum (dual norm) and argmax vector u from
% max_u (u, v) s.t. \sqrt{\norm{u}_2^2 + (1/s) \norm{u}_1^2} \leq 1

[p, r] = size(v);
vinf = vecnorm(v, 'inf', 1);

% The formula is
% inf_{v = v_1 + v_2} \sqrt{\norm{v_1}_2^2 + s \norm{v_2}_\infty^2}.
% v_1 is a soft-thresholded v with threshold a satisfying \norm{v_1}_1 = s*a.
% Then u is proportional to v_1

% Newton search over a
a = zeros(1, r);
converged = false;
while ~converged
    v1 = sign(v) .* max(abs(v) - a, 0);
    nd1 = s*a - vecnorm(v1, 1, 1);
    nd2 = s + sum(v1 ~= 0, 1);
    
    a = a - nd1./nd2;
    
    converged = all(abs(nd1) <= 1e-10*vinf);
end
n = sqrt(sum(v1.^2, 1) + s*a.^2);
u = v1 ./ thetasnorm(v1, s);

end


