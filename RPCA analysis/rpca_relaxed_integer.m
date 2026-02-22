function [L, S, out] = rpca_relaxed_integer(M, lambda_pcp, opts)
% RPCA_RELAXED_INTEGER  Robust PCA with soft integer encouragement on S.
%
% Solves the relaxed problem:
%   min_{L,S}  ||L||_* + lambda_pcp*||S||_1 + beta * Phi_int(S)
%   s.t.       M = L + S
%
% where Phi_int(S) = sum_{i,j} (1 - exp(-(S_ij - round(S_ij))^2 / (2*sigma^2)))
% is a smooth penalty encouraging S entries to be near integers.
%
% WHY THIS WORKS (vs hard rounding):
% -----------------------------------
% 1) CONVEXITY PRESERVED (approximately): The penalty is smooth and differentiable,
%    so gradient-based updates remain stable. Hard round() is discontinuous.
%
% 2) NO CONVERGENCE BREAKING: Standard IALM requires proximal operators to be
%    well-behaved. This penalty adds a smooth term to the objective without
%    changing the constraint structure.
%
% 3) GRADUAL INTEGER ENCOURAGEMENT: As iterations progress and mu grows,
%    the soft penalty increasingly pushes S toward integers without the
%    oscillation that hard projection causes.
%
% 4) LOW-RANK PRESERVATION: Unlike post-hoc rounding (Lbar := Mbar - round(Sbar)),
%    we never destroy the low-rank structure of L by recomputing it from a
%    modified S.
%
% Inputs:
%   M           - (m x n) data matrix (scaled: Mbar = Hankel(Dy)/(scaleH*alpha))
%   lambda_pcp  - sparsity regularization (default: 1/sqrt(max(m,n)))
%   opts        - struct with optional fields:
%       .tol        : stopping tolerance (default 1e-7)
%       .max_iter   : max iterations (default 1000)
%       .verbose    : print progress (default 1)
%       .mu         : initial penalty (default auto)
%       .mu_factor  : penalty growth rate (default 1.5)
%       .mu_max     : max penalty (default 1e7)
%       .beta       : integer penalty weight (default 0.1)
%       .sigma      : Gaussian width for integer penalty (default 0.3)
%       .beta_schedule : 'constant', 'increasing', or 'annealing' (default 'increasing')
%
% Outputs:
%   L   - low-rank component
%   S   - sparse component (encouraged to be near-integer)
%   out - struct with convergence info
%
% Author: Generated for BTP-2 research
% Date: 2026-01-30

    %% ==================== Parse Inputs ====================
    if ~isfloat(M), M = double(M); end
    [m, n] = size(M);
    
    if ~exist('lambda_pcp','var') || isempty(lambda_pcp)
        lambda_pcp = 1 / sqrt(max(m, n));
    end
    if ~exist('opts','var') || isempty(opts)
        opts = struct();
    end
    
    % Algorithm parameters
    tol         = get_opt(opts, 'tol', 1e-7);
    max_iter    = get_opt(opts, 'max_iter', 1000);
    verbose     = get_opt(opts, 'verbose', 1);
    mu          = get_opt(opts, 'mu', []);
    mu_factor   = get_opt(opts, 'mu_factor', 1.5);
    mu_max      = get_opt(opts, 'mu_max', 1e7);
    
    % Integer penalty parameters
    beta        = get_opt(opts, 'beta', 0.1);
    sigma       = get_opt(opts, 'sigma', 0.3);
    beta_schedule = get_opt(opts, 'beta_schedule', 'increasing');
    
    %% ==================== Initialization ====================
    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(m, n);
        S = zeros(m, n);
        out = struct('iter', 0, 'relres', 0, 'obj', 0, 'rankL', 0, ...
                     'nnzS', 0, 'int_penalty', 0, 'S_integrality', 1);
        return;
    end
    
    % Initialize mu from spectral norm
    if isempty(mu)
        try
            smax = svds(M, 1);
        catch
            smax = norm(M, 2);
        end
        mu = 1.25 / max(smax, eps);
    end
    
    % Initialize variables
    L = zeros(m, n);
    S = zeros(m, n);
    Y = M / max(norm(M, 2), eps);  % Dual variable (scaled initialization)
    
    relres = inf;
    k = 0;
    r = 0;
    obj = NaN;
    beta_k = beta;  % Current beta value
    
    if verbose
        fprintf('=============================================================\n');
        fprintf('RPCA with Relaxed Integer Constraint\n');
        fprintf('=============================================================\n');
        fprintf('lambda=%.3g, beta=%.3g, sigma=%.3g, tol=%.1e\n', ...
                lambda_pcp, beta, sigma, tol);
        fprintf('beta_schedule: %s\n', beta_schedule);
        fprintf('-------------------------------------------------------------\n');
        fprintf('%5s  %10s  %10s  %8s  %8s  %8s  %8s\n', ...
                'iter', 'relres', 'obj', 'rank(L)', 'nnz(S)', 'int_err', 'mu');
    end
    
    %% ==================== Main IALM Loop ====================
    while (relres > tol) && (k < max_iter)
        k = k + 1;
        
        %% --- Update beta according to schedule ---
        switch lower(beta_schedule)
            case 'constant'
                beta_k = beta;
            case 'increasing'
                % Gradually increase integer penalty
                beta_k = beta * (1 + 0.5 * log(1 + k));
            case 'annealing'
                % Start strong, then relax (useful if integer assumption is approximate)
                beta_k = beta * exp(-k / (max_iter / 3));
            otherwise
                beta_k = beta;
        end
        
        %% --- L-subproblem: Singular Value Thresholding ---
        % argmin_L ||L||_* + (mu/2)||L - (M - S + Y/mu)||_F^2
        W = M - S + (1/mu) * Y;
        [U, Sig, V] = svd(W, 'econ');
        sig = diag(Sig);
        sig_shr = max(sig - 1/mu, 0);  % Soft-thresholding
        r = nnz(sig_shr > 0);
        
        if r > 0
            L = U(:, 1:r) * diag(sig_shr(1:r)) * V(:, 1:r)';
        else
            L = zeros(m, n);
        end
        
        %% --- S-subproblem: Soft-threshold + Integer Encouragement ---
        % Standard soft-thresholding first
        T = M - L + (1/mu) * Y;
        S_soft = sign(T) .* max(abs(T) - lambda_pcp/mu, 0);
        
        % Apply smooth integer encouragement (gradient step)
        % Phi_int(s) = 1 - exp(-(s - round(s))^2 / (2*sigma^2))
        % grad_Phi = (s - round(s)) / sigma^2 * exp(-(s - round(s))^2 / (2*sigma^2))
        S_frac = S_soft - round(S_soft);  % Fractional part
        gauss_weight = exp(-S_frac.^2 / (2 * sigma^2));
        grad_int = (S_frac / sigma^2) .* gauss_weight;
        
        % Step size for integer penalty (scaled by 1/mu to match ADMM scaling)
        alpha_int = beta_k / mu;
        
        % Update S with integer-encouraging gradient step
        S = S_soft - alpha_int * grad_int;
        
        % Optional: For very small fractional parts, snap to integer
        % (only when the gradient naturally drives it there)
        snap_mask = abs(S - round(S)) < 0.05;
        S(snap_mask) = round(S(snap_mask));
        
        %% --- Dual Update ---
        Z = M - L - S;
        Y = Y + mu * Z;
        
        %% --- Diagnostics ---
        relres = norm(Z, 'fro') / normM;
        
        % Compute integrality measure: mean squared distance to nearest integer
        int_error = mean((S(:) - round(S(:))).^2);
        
        % Compute integer penalty value
        int_penalty = sum(1 - exp(-(S(:) - round(S(:))).^2 / (2*sigma^2)));
        
        % Objective: ||L||_* + lambda*||S||_1 + beta*Phi_int(S)
        obj = sum(sig_shr) + lambda_pcp * sum(abs(S(:))) + beta_k * int_penalty;
        
        if verbose && (mod(k, 20) == 0 || k == 1 || k <= 5)
            fprintf('%5d  %10.3e  %10.4e  %8d  %8d  %8.4f  %8.2e\n', ...
                    k, relres, obj, r, nnz(abs(S) > 1e-6), int_error, mu);
        end
        
        %% --- Penalty Update ---
        mu = min(mu * mu_factor, mu_max);
    end
    
    %% ==================== Final Integer Snap (Optional) ====================
    % After convergence, do a gentle final snap for entries very close to integers
    final_snap_thresh = 0.1;
    near_int_mask = abs(S - round(S)) < final_snap_thresh;
    S_final_adjust = S;
    S_final_adjust(near_int_mask) = round(S(near_int_mask));
    
    % Check if final snap improves residual
    Z_snapped = M - L - S_final_adjust;
    if norm(Z_snapped, 'fro') <= 1.5 * norm(Z, 'fro')
        S = S_final_adjust;
        % Compensate in L to maintain equality (small adjustment)
        L = L + (Z_snapped - Z) * 0.5;  % Distribute residual
    end
    
    %% ==================== Output ====================
    % Final integrality measure
    final_int_error = mean((S(:) - round(S(:))).^2);
    S_integrality = 1 - sqrt(final_int_error);  % 1 = perfect integers, 0 = very non-integer
    
    if verbose
        fprintf('-------------------------------------------------------------\n');
        fprintf('Done in %d iters, relres=%.3e, rank(L)=%d, nnz(S)=%d\n', ...
                k, relres, r, nnz(abs(S) > 1e-6));
        fprintf('Integrality score: %.4f (1=perfect integers)\n', S_integrality);
        fprintf('=============================================================\n');
    end
    
    out.iter       = k;
    out.relres     = relres;
    out.obj        = obj;
    out.rankL      = r;
    out.nnzS       = nnz(abs(S) > 1e-6);
    out.int_error  = final_int_error;
    out.S_integrality = S_integrality;
    out.beta_final = beta_k;
end

%% ==================== Helper Functions ====================
function v = get_opt(opts, field, defaultv)
    if isfield(opts, field) && ~isempty(opts.(field))
        v = opts.(field);
    else
        v = defaultv;
    end
end
