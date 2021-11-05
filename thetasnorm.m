function n = thetasnorm(u, s)
    n = sqrt(sum(u.^2, 1) + (1/s)*sum(abs(u), 1).^2);
end

