function [Unew, Vnew] = balance_UV(U, V, s, tol)
    U_norms_ = thetasnorm(U, s);
    V_norms_ = thetasnorm(V, s);
    z_ = U_norms_.*V_norms_ >= tol;
    
    if any(z_)
        scaling = sqrt(U_norms_(z_) ./ V_norms_(z_));
        Unew = U(:, z_) ./ scaling;
        Vnew = V(:, z_) .* scaling;
    else
        Unew = zeros(size(U, 1), 1);
        Vnew = Unew;
end