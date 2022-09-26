function [Uhat, Vhat, fval] = pr_opt_altmin(X, Y, lambda, s)

[n, p] = size(X);

% Play with these
opttol = 1e-4;
altmin_tol = 1e-3;

normsqest = 1/sqrt(n) + max(mean(Y), 0);

% Normalize
Y = Y/normsqest;
lambda = lambda/normsqest;

% thetastol = normsqest*opttol;

[n, p] = size(X);

    function fbval = fb(U_, V_)
        fbval = (0.5/n)*sum((Y - sum((X*U_).*(X*V_), 2)).^2);
    end


reg_ind = @(U_, V_) lambda*thetasnorm(U_, s).*thetasnorm(V_, s);
reg_f = @(U_, V_) sum(reg_ind(U_, V_));

f = @(U_, V_) fb(U_, V_) + reg_f(U_, V_);

    function U_ = sq_lasso_solve(XV_, Vnorms, U0_)
    converged = false;
        function [loss_, g_] = loss_grad_fun_(U__)
            XU__ = X*U__;
            r = Y - sum(XV_.*XU__, 2);
            
            g_ = -(1/n)* ((r.*XV_)'*X)'; % = -(1/n)*X'*(r.*XV_)
            loss_ = (1/(2*n))*sum(r.^2);
        end
        
        function loss_ = loss_fun_(U__)
            r = Y - sum(XV_.*(X*U__), 2);
            loss_ = (1/(2*n))*sum(r.^2);
        end

    U_ = U0_;
    i_ = 0;
    t = 1;
    U_old = U_;
    y_ = U_;
    
    stepsize_ = 1;
    while ~converged
        [loss_y_, g_y_] = loss_grad_fun_(y_);
        backtracking = true;
        while backtracking
            stepsizes = stepsize_ ./ Vnorms;
            U_new_ = prox_l2_l1(stepsizes.*g_y_, y_, lambda*stepsize_, lambda*stepsize_/sqrt(s));
            loss_new = loss_fun_(U_new_);
            if loss_new <= loss_y_ + sum((U_new_ - y_).*g_y_, 'all') + sum(sum((U_new_ - y_).^2, 1)./(2*stepsizes)) + 1e-10
                backtracking = false;
                U_ = U_new_;
                stepsize_ = 1.2*stepsize_;
            else
                stepsize_ = 0.8*stepsize_;
            end
        end
        
        % Check optimality conditions
        if mod(i_, 5) == 4
            nz_col = thetasnorm(U_, s).*Vnorms >= opttol;
            U_nz_ = U_(:, nz_col);

            % Gradient including L2 term, normalized by L1 penalty
            [~, g_U_] = loss_grad_fun_(U_); 
            gl2_norm = sqrt(s) * (g_U_(:, nz_col) ./ (lambda*Vnorms(nz_col)) + U_nz_ ./ vecnorm(U_nz_, 2, 1));

            nz_ind = find(U_nz_);
            err1 = max(gl2_norm(nz_ind) + sign(U_nz_(nz_ind)));

            err2 = max(abs(gl2_norm(:))) - 1;

            if max(err1, err2) <= altmin_tol
                % Check zero columns subgradient condition
                if any(~nz_col)
                    gl2_zero = g_U_(:, ~nz_col) ./ (lambda * Vnorms(~nz_col));
                    gl2_z_thresh = max(abs(gl2_zero) - 1/sqrt(s), 0);
                    if all(vecnorm(gl2_z_thresh, 2, 1) <= 1 + altmin_tol)
                        converged = true;
                    end
                else
                    converged = true;
                end
            end
        end
        
        % Adaptive restart
        if sum(g_y_.*(U_ - U_old), 'all') > 0
            t = 1;
            y_ = U_;
        else
            t_old = t;
            t = (1 + sqrt(1 + 4*t^2))/2;
            y_ = U_ + ((t_old - 1)/t)*(U_ - U_old);
        end
        
        U_old = U_;
        i_ = i_ + 1;
    end
    end

    % Variables to keep track of whether optimization is stalling
    last_best_ = inf;
    count_since_best_ = 0;
    
    function stop = opt_cond_(U_, V_)
        XUV = (X*U_).*(X*V_);
        alpha_ = Y - sum(XUV, 2);
        optcond_lhs = (1/n)*sum(alpha_.*XUV, 1);
        optcond_rhs = reg_ind(U_, V_);
        
        max_err = max(abs(optcond_lhs - optcond_rhs));
        if max_err < 0.99*last_best_
            last_best_ = max_err;
            count_since_best_ = 0;
        else
            count_since_best_ = count_since_best_ + 1;
        end
        
        % Play with this
        stop = (max_err <= lambda*opttol) || (count_since_best_ >= 10);
    end

% Spectral initialization
Xsqr = X.^2;
[~, ii] = sort(Y'*Xsqr, 'descend');
ii = ii(1:s);
Z_ii = (1/n)*X(:, ii)'*(Y.*X(:, ii));
[V_ii, D_ii] = eigs((Z_ii + Z_ii')/2, 1, 'largestreal');

U = zeros(p, 1);
U(ii) = sqrt(D_ii/3)*V_ii;
V = U;

done = false;
while ~done
    
    fprintf('Solution rank: %d\n', size(U, 2));
    
    altmin_conv = false;
    altmin_it = 0;
    while ~altmin_conv
        XU = X*U;
        V = sq_lasso_solve(XU, thetasnorm(U, s), V);
        [U, V] = balance_UV(U, V, s, opttol);
        XV = X*V;
        U = sq_lasso_solve(XV, thetasnorm(V, s), U);
        [U, V] = balance_UV(U, V, s, opttol);
        
        altmin_conv = opt_cond_(U, V);
        altmin_it = altmin_it + 1;
    end
    
    fval = f(U, V);
    res = Y - sum((X*U).*(X*V), 2);
    Z = (1/n)*(res.*X)'*X;
    [M, I] = max(abs(Z(:)));
    [maxI, maxJ] = ind2sub([p, p], I);
    
    improvement = false;
    if M > lambda*((1 + 1/sqrt(s))^2 + opttol)
        U_new = [U, zeros(p, 1)];
        V_new = [V, zeros(p, 1)];
        h = 0.1;
        % Backtracking
        for i = 1:50
            U_new(maxI, end) = sqrt(h);
            V_new(maxJ, end) = sqrt(h)*sign(Z(maxI, maxJ));
            fval_new = f(U_new, V_new);
            if fval_new < fval - lambda*opttol
                improvement = true;
                fval = fval_new;
                U = U_new;
                V = V_new;
                break;
            end
            h = h/2;
        end
    end
    
    done = ~improvement;
end

Uhat = sqrt(normsqest)*U;
Vhat = sqrt(normsqest)*V;
fval = fval*normsqest^2;

end


