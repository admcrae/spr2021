function n = thetasnorm(u, s)
    n = vecnorm(u, 2, 1) + (1 / sqrt(s))*vecnorm(u, 1, 1);
end

