function [L,S,out] = rpca_pcp(M, lambda, opts)
% RPCA via Principal Component Pursuit (PCP) using Inexact ALM.
%   Solve:  min_{L,S} ||L||_* + lambda * ||S||_1   s.t.  L + S = M
%
% Inputs:
%   M       - (m x n) data matrix (double)
%   lambda  - positive scalar (optional). Default: 1/sqrt(max(m,n))
%   opts    - struct with fields (all optional):
%               tol        : stopping tolerance on relative residual (default 1e-7)
%               max_iter   : max iterations (default 1000)
%               mu         : initial penalty parameter (default 1.25/svds(M,1))
%               mu_factor  : penalty growth factor (default 1.5)
%               mu_max     : maximum mu (default 1e7)
%               verbose    : 0/1 print progress (default 1)
%
% Outputs:
%   L       - low-rank component
%   S       - sparse component
%   out     - struct: fields iter, relres, obj, rankL, nnzS (per iteration last)
%
%  - Uses soft-thresholding on singular values and entries.
%  - Stopping rule: ||M-L-S||_F / ||M||_F <= tol

    % Defaults
    if ~isfloat(M), M = double(M); end
    [m,n] = size(M);
    if ~exist('lambda','var') || isempty(lambda)
        lambda = 1/sqrt(max(m,n));
    end
    if ~exist('opts','var') || isempty(opts), opts = struct(); end
    tol       = get_opt(opts,'tol',      1e-7);
    max_iter  = get_opt(opts,'max_iter', 1000);
    verbose   = get_opt(opts,'verbose',  1);
    mu        = get_opt(opts,'mu',       []);
    mu_factor = get_opt(opts,'mu_factor',1.5);
    mu_max    = get_opt(opts,'mu_max',   1e7);

    % Initialize
    normM = norm(M,'fro'); if normM==0, L=zeros(m,n); S=zeros(m,n); out=struct('iter',0,'relres',0,'obj',0,'rankL',0,'Sparsity',0); return; end
    if isempty(mu)
        % spectral norm (largest singular value)
        try
            smax = svds(M,1);
        catch
            smax = norm(M,2);
        end
        mu = 1.25 / max(smax, eps);
    end

    L = zeros(m,n);
    S = zeros(m,n);
    Y = M / max(norm(M,2), eps);  % dual variable initialization (scaled)

    relres = inf;
    k = 0;
    if verbose
        fprintf('RPCA-PCP (IALM) | lambda=%.3g, tol=%.1e\n', lambda, tol);
        fprintf('%5s  %10s  %10s  %8s  %8s  %8s\n','iter','relres','obj','rank(L)','nnz(S)','mu');
    end

    while (relres > tol) && (k < max_iter)
        k = k + 1;

        % === Singular value thresholding step (update L) ===
        %   L = argmin_L ||L||_* + (mu/2)||L - (M - S + (1/mu)Y)||_F^2
        W = M - S + (1/mu) * Y;
        [U,Sigma,V] = svd(W,'econ');
        sigma = diag(Sigma);
        sigma_shr = soft(sigma, 1/mu);
        r = sum(sigma_shr > 0);
        if r > 0
            L = U(:,1:r) * diag(sigma_shr(1:r)) * V(:,1:r)';
        else
            L = zeros(m,n);
        end

        % === Element-wise soft-thresholding (update S) ===
        %   S = argmin_S lambda||S||_1 + (mu/2)||S - (M - L + (1/mu)Y)||_F^2
        T = M - L + (1/mu) * Y;
        S = soft(T, lambda/mu);

        % === Dual update ===
        Z = M - L - S;
        Y = Y + mu * Z;

        % === Diagnostics ===
        relres = norm(Z,'fro') / normM;
        obj = sum(sigma_shr) + lambda * sum(abs(S(:)));
        if verbose && (mod(k,10)==0 || k==1)
            fprintf('%5d  %10.3e  %10.4e  %8d  %8d  %8.2e\n', k, relres, obj, r, nnz(abs(S)>0), mu);
        end

        % === Penalty growth ===
        mu = min(mu * mu_factor, mu_max);
    end

    if verbose
        fprintf('Done in %d iters, relres=%.3e, rank(L)=%d, nnz(S)=%d\n', k, relres, r, nnz(abs(S)>0));
    end

    out.iter   = k;
    out.relres = relres;
    out.obj    = obj;
    out.rankL  = r;
    out.nnzS   = nnz(abs(S)>0);

end

% ---------- helpers ----------
function v = get_opt(opts, field, defaultv)
    if isfield(opts,field) && ~isempty(opts.(field)), v = opts.(field); else, v = defaultv; end
end

function Xs = soft(X, tau)
% Soft-thresholding (entrywise or vector of singular values)
    Xs = sign(X) .* max(abs(X) - tau, 0);
end
