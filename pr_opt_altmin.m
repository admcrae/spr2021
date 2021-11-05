function [Uhat, Vhat, fval] = pr_opt_altmin(X, Y, lambda, s)

[n, p] = size(X);

% Play with these
opttol = 3e-5;
altmin_tol = 1e-2;

normsqest = 1/sqrt(n) + max(mean(Y), 0);

[n, p] = size(X);

    function fbval = fb(U_, V_)
        fbval = (0.5/n)*sum((Y - sum((X*U_).*(X*V_), 2)).^2);
    end


reg_single = @(U_) 0.5*lambda*(vecnorm(U_, 2, 1).^2 +(1/s)*vecnorm(U_, 1, 1).^2);
reg_ind = @(U_, V_) reg_single(U_) + reg_single(V_);
reg_f = @(U_, V_) sum(reg_ind(U_, V_));

f = @(U_, V_) fb(U_, V_) + reg_f(U_, V_);

    % Utility function for computing prox-map on all columns of U or V
    function u_ = prox_map(grad_, x_, step_)
        u_ = zeros(size(x_));
        for k = 1:size(x_, 2)
            u_(:, k) = prox_l1(step_*grad_(:, k), x_(:, k), step_*lambda, step_*lambda/s);
        end
    end

    function U_ = sq_lasso_solve(XV_, U0_)
    % Solve modified LASSO
    % argmin_u (1/(2*n))*( \sum_{i=1}^n (Y - \sum_{k=1}^r (Xm(i, :, k) ,u_k)) )^2
    %       + 0.5*\sum_{k=1}^r (||u_k||_2^2 + (lambda/s)*||u_k||_1^2 )
    % by accelerated proximal method

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
    
    % Initial stepsize uses estimated Lipschitz constant based on size
    % of observations. This may need to be adjusted for design vectors 
    % that are not unit-variance
    stepsize_ = 0.1*n / sum(XV_.^2, 'all');
    while ~converged
        [loss_y_, g_y_] = loss_grad_fun_(y_);
        
        backtracking = true;
        while backtracking
            U_new_ = prox_map(g_y_, y_, stepsize_);
            loss_new = loss_fun_(U_new_);
            if loss_new <= loss_y_ + sum((U_new_ - y_).*g_y_, 'all') + sum((U_new_ - y_).^2, 'all')/(2*stepsize_) + 1e-10
                backtracking = false;
                U_ = U_new_;
                stepsize_ = 1.05*stepsize_;
            else
                stepsize_ = 0.8*stepsize_;
            end
        end
        
        % Check optimality conditions
        if mod(i_, 5) == 4
            % Gradient including L2 term, normalized by L1 penalty
            [~, g_U_] = loss_grad_fun_(U_); 
            gl2_norm = (g_U_ + lambda*U_)./(lambda/s*vecnorm(U_, 1, 1));

            nz_ind = find(U_(:));
            err1 = max(gl2_norm(nz_ind) + sign(U_(nz_ind)));
            err2 = max(abs(gl2_norm(:))) - 1;

            if max(err1, err2) <= altmin_tol
                converged = true;
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
        stop = (max_err <= lambda*opttol*normsqest) || (count_since_best_ >= 40);
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

    % Utility function to make colums of U_, V_ have equal \theta_s norms
    function [Unew_, Vnew_] = balance_UV_(U_, V_)
        Unew_ = zeros(size(U_));
        Vnew_ = zeros(size(V_));
        U_norms_ = thetasnorm(U_, s);
        V_norms_ = thetasnorm(V_, s);
        z_ = (U_norms_ == 0) | (V_norms_ == 0);
        
        scaling = sqrt(U_norms_(~z_) ./ V_norms_(~z_));
        Unew_(:, ~z_) = U_(:, ~z_) ./ scaling;
        Vnew_(:, ~z_) = V_(:, ~z_) .* scaling;
    end

done = false;
while ~done
    
    fprintf('Solution rank: %d\n', size(U, 2));
    
    altmin_conv = false;
    altmin_it = 0;
    while ~altmin_conv
        XU = X*U;
        V = sq_lasso_solve(XU, V);
        [U, V] = balance_UV_(U, V);
        XV = X*V;
        U = sq_lasso_solve(XV, U);
        [U, V] = balance_UV_(U, V);
        
        altmin_conv = opt_cond_(U, V);
        altmin_it = altmin_it + 1;
    end
    
    fval = f(U, V);
    res = Y - sum((X*U).*(X*V), 2);
    Z = (1/n)*(res.*X)'*X;
    [M, I] = max(abs(Z(:)));
    [maxI, maxJ] = ind2sub([p, p], I);
    
    improvement = false;
    if M > lambda*((1 + 1/s) + opttol)
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

Uhat = U;
Vhat = V;

end


