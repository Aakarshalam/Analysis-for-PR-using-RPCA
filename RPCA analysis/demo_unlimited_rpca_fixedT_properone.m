function demo_unlimited_rpca_fixedT()
% Fixed-time RPCA recovery under folding with a high-rate "continuous" model.
% 1) Build a high-rate multitone x_cont(t) at FS_CONT.
% 2) Fold x_cont(t) into [-lambda,lambda) at high rate.
% 3) Sample both x_cont and folded version at a lower rate fs (baseline fs0).
% 4) Perform Hankel(Δy) + RPCA at fs0 (sweep Hankel height L).
% 5) Sweep fs with fixed T, sampling from the same high-rate x_cont and
%    running the same RPCA pipeline to get error/SNR vs fs.

rng(0);

%% -------- 0) User params (edit here) --------
T        = 0.02;        % total duration [s] (fixed)
FS_CONT  = 200e3;       % high-rate "continuous" proxy [Hz]

fs0      = 7000;        % baseline sampling rate for RPCA [Hz]
lambda   = 1.5;         % folding threshold (range [-lambda, lambda])

% Underlying (continuous-time) multitone
A1  = 2.0;  f1  = 120;  phi1 = 0.30*pi;
A2  = 1.2;  f2  = 300;  phi2 = -0.55*pi;

% RPCA/IALM options
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%% -------- 1) Build high-rate continuous signal and fold it --------
t_cont = 0:1/FS_CONT:T-1/FS_CONT;

x_cont = A1*cos(2*pi*f1*t_cont + phi1) + ...
         A2*cos(2*pi*f2*t_cont + phi2);

x_cont_fold = fold_centered(x_cont, lambda);

% High-rate visualization
figure('Name','High-rate original vs folded','Color','w');
plot(t_cont, x_cont, 'LineWidth',1.0); hold on;
plot(t_cont, x_cont_fold, '--', 'LineWidth',1.0);
grid on; xlim([0, T]);
xlabel('t [s]'); ylabel('amplitude');
title(sprintf('High-rate signal @ FS_{CONT}=%.0f Hz (\\lambda=%.2f)', FS_CONT, lambda));
legend('x_{cont}(t)','folded x_{cont}(t)','Location','best');

%% -------- 2) Baseline low-rate sampling at fs0 from high-rate --------
t0 = 0:1/fs0:T-1/fs0;                  % actual sampling instants
N0 = numel(t0);
idx0 = round(t0 * FS_CONT) + 1;        % indices into high-rate grid
idx0 = min(max(idx0,1), numel(t_cont));

x0 = x_cont(idx0);                     % true samples at fs0
y0 = x_cont_fold(idx0);                % folded samples at fs0

% Show sampled original vs folded (baseline)


% Wrap-density diagnostics (baseline)
ktrue0 = round((x0 - y0)/(2*lambda));
wraps0 = find(diff(ktrue0)~=0);
fprintf('[BASE] fs0=%.0f Hz: wraps=%d  (%.2f%% of samples)\n', ...
        fs0, numel(wraps0), 100*numel(wraps0)/N0);

%% -------- 3) Hankel(Δy0) + RPCA at fs0 (L sweep) --------
dy0 = diff(y0);             % first difference of folded samples
m0  = numel(dy0);

% Pick a range of Hankel heights L
L_list = unique(round([0.25 0.35 0.45 0.5 0.6 0.7 0.8] * m0));
L_list(L_list < 20) = [];
L_list(L_list > m0-20) = [];

snr_list = nan(size(L_list));
best = struct('snr', -inf);

fprintf('\n--- RPCA on Hankel(Δy) at fs0: sweeping L ---\n');
for ii = 1:numel(L_list)
    L = L_list(ii);
    H = hankel_from_vector(dy0, L);

    % Normalize to stabilize PCP
    scaleH = median(abs(H(:)) + eps);
    Hn = H / scaleH;

    % PCP on Hn
    lam_pcp = 1/sqrt(max(size(Hn)));
    [HLn, ~, out] = rpca_pcp(Hn, lam_pcp, opts);
    HL = HLn * scaleH;

    % De-Hankelize -> Δx̂, integrate -> x̂, then 2λ branch alignment
    dy_hat = dehankel_to_vector(HL);
    x_hat  = [y0(1), y0(1) + cumsum(dy_hat)];
    kstar  = median( round( (x_hat - y0)/(2*lambda) ) );
    x_hat  = x_hat - 2*lambda*kstar;

    % Quality (SNR vs ground truth x0)
    snr_list(ii) = 20*log10( max(norm(x0),eps) / max(norm(x0 - x_hat),eps) );
    fprintf('  L=%4d | %dx%d | relres=%.2e | SNR=%.2f dB\n', ...
            L, size(H,1), size(H,2), out.relres, snr_list(ii));

    if snr_list(ii) > best.snr
        best = struct('L',L,'x_hat',x_hat,'dy_hat',dy_hat,'HL',HL,'snr',snr_list(ii));
    end
end
figure('Name','Original vs Folded (fixed T, fs0 samples)','Color','w');
subplot(3,1,1);
plot(t0, x0, 'LineWidth',1.1); grid on;
xlabel('t [s]'); ylabel('amplitude');
title(sprintf('Original x[n] from high-rate model | T=%.3fs, fs=%.0fHz, N=%d', ...
      T, fs0, N0));

subplot(3,1,2);
plot(t0, y0, 'LineWidth',1.1); grid on;
xlabel('t [s]'); ylabel('amplitude');
title(sprintf('Folded y[n] in [ -\\lambda, \\lambda ], \\lambda=%.2f', lambda));

subplot(3,1,3);
% Plot best recovery at fs0

plot(t0, x0, 'LineWidth',1.25); hold on;
plot(t0, best.x_hat, '--', 'LineWidth',1.25); grid on;
xlabel('t [s]'); ylabel('amplitude');
title(sprintf('Best recovery at fs=%.0f Hz | L=%d | SNR=%.2f dB', ...
      fs0, best.L, best.snr));
legend('x (true samples)','\hat{x} (RPCA)','Location','best');

% Plot SNR vs L
figure('Name','Effect of Hankel height (fs0)','Color','w');
plot(L_list, snr_list, 'o-','LineWidth',1.25); grid on;
xlabel('Hankel height L'); ylabel('SNR (dB)');
title('Recovery SNR vs L (Δy, fixed T, baseline fs)');

%% -------- 4) Sweep fs with fixed T (N varies; sampling from x_cont) --------
fs_min = 240; 
fs_max = 1e4; 
npts   = 100;
fs_vec = round(logspace(log10(fs_min), log10(fs_max), npts));

err_rel  = nan(size(fs_vec));
snr_db   = nan(size(fs_vec));
wrap_pct = nan(size(fs_vec));

fprintf('\n--- Sweep fs with fixed T=%.3fs (sampling from x_{cont}) ---\n', T);

for kk = 1:numel(fs_vec)
    fs_k = fs_vec(kk);

    % Sample times at fs_k over [0,T)
    t_k   = 0:1/fs_k:T-1/fs_k;
    N_k   = numel(t_k);
    idx_k = round(t_k * FS_CONT) + 1;
    idx_k = min(max(idx_k,1), numel(t_cont));

    % Samples (clean & folded) from the same continuous model
    x_k = x_cont(idx_k);
    y_k = x_cont_fold(idx_k);

    % Wrap density at fs_k
    ktrue_k = round((x_k - y_k)/(2*lambda));
    wraps_k = find(diff(ktrue_k)~=0);
    wrap_pct(kk) = 100*numel(wraps_k)/N_k;

    % Hankel(Δy_k)
    dy_k = diff(y_k);
    m_k  = numel(dy_k);
    if m_k < 40, continue; end          % not enough samples for meaningful Hankel

    L_k = floor(0.5*m_k);               % near-balanced Hankel height
    H_k = hankel_from_vector(dy_k, L_k);

    % Normalize
    scaleH = median(abs(H_k(:)) + eps);
    Hn_k   = H_k / scaleH;

    % PCP
    lam_pcp = 1/sqrt(max(size(Hn_k)));
    [HLn_k, ~, outk] = rpca_pcp(Hn_k, lam_pcp, opts); 
    HL_k = HLn_k * scaleH;

    % Back to Δx̂, integrate, and align branches
    dy_hat_k = dehankel_to_vector(HL_k);
    x_hat_k  = [y_k(1), y_k(1) + cumsum(dy_hat_k)];
    kstar_k  = median( round( (x_hat_k - y_k)/(2*lambda) ) );
    x_hat_k  = x_hat_k - 2*lambda*kstar_k;

    % Metrics vs clean x_k
    err_rel(kk) = norm(x_k - x_hat_k) / max(norm(x_k), eps);
    snr_db(kk)  = 20*log10( max(norm(x_k),eps) / max(norm(x_k - x_hat_k),eps) );
end

%% -------- 5) Plots: error & wrap density vs fs, SNR vs fs --------
figure('Name','Error vs Sampling (fixed T, sampling from x_{cont})','Color','w');

subplot(1,2,1);
semilogx(fs_vec, err_rel, 'o-','LineWidth',1.25); grid on; hold on;
yyaxis right;
semilogx(fs_vec, wrap_pct, 's--','LineWidth',1.1);
ylabel('Wrap density (%)');
yyaxis left;
xlabel('Sampling frequency f_s (Hz)');
ylabel('Relative error  ||x-\hat{x}||/||x||');
title(sprintf('Fixed T=%.2fs: error & wrap density vs f_s', T));
legend('Rel. error','Wrap density','Location','best');

subplot(1,2,2);
semilogx(fs_vec, snr_db, 'o-','LineWidth',1.25); grid on;
xlabel('Sampling frequency f_s (Hz)');
ylabel('SNR (dB)');
title('Fixed T: recovery SNR vs f_s (RPCA on Hankel(Δy))');

end

%% ================= helpers =================
function y = fold_centered(x, lambda)
% Amplitude folding into [-lambda, +lambda)
    y = mod(x + lambda, 2*lambda) - lambda;
end

function H = hankel_from_vector(v, L)
% Build Hankel matrix from a vector v with height L
    N = numel(v); 
    M = N - L + 1;
    if M <= 0, error('Hankel height L too large for length %d.', N); end
    H = hankel(v(1:L), v(L:N));
end

function v = dehankel_to_vector(H)
% Anti-diagonal averaging back to a 1-D sequence
    [L,M] = size(H); 
    Lm = L + M - 1;
    v = zeros(1, Lm); 
    c = zeros(1, Lm);
    for i = 1:L
        for j = 1:M
            k = i + j - 1;
            v(k) = v(k) + H(i,j);
            c(k) = c(k) + 1;
        end
    end
    v = v ./ c;
end

% ---------- RPCA (PCP via IALM) ----------
function [L,S,out] = rpca_pcp(M, lambda, opts)
    if ~isfloat(M), M = double(M); end
    [m,n] = size(M);
    if ~exist('lambda','var') || isempty(lambda), lambda = 1/sqrt(max(m,n)); end
    if ~exist('opts','var') || isempty(opts), opts = struct(); end
    tol       = get_opt(opts,'tol',1e-7);
    max_iter  = get_opt(opts,'max_iter',1000);
    verbose   = get_opt(opts,'verbose',1);
    mu        = get_opt(opts,'mu',[]);
    mu_factor = get_opt(opts,'mu_factor',1.5);
    mu_max    = get_opt(opts,'mu_max',1e7);

    normM = norm(M,'fro'); 
    if normM==0
        L=zeros(m,n); S=zeros(m,n);
        out=struct('iter',0,'relres',0,'obj',0,'rankL',0,'nnzS',0);
        return;
    end
    if isempty(mu)
        try
            smax = svds(M,1);
        catch
            smax = norm(M,2);
        end
        mu = 1.25 / max(smax, eps);
    end

    L = zeros(m,n); 
    S = zeros(m,n); 
    Y = M / max(norm(M,2), eps);
    relres = inf; 
    k      = 0; 
    r      = 0; 
    obj    = NaN;

    if verbose
        fprintf('RPCA-PCP: lambda=%.3g, tol=%.1e\n', lambda, tol);
        fprintf('%5s  %10s  %10s  %8s  %8s  %8s\n',...
                'iter','relres','obj','rank(L)','nnz(S)','mu');
    end

    while (relres > tol) && (k < max_iter)
        k = k + 1;

        % --- low-rank update (SVT) ---
        W = M - S + (1/mu)*Y;
        [U,Sig,V] = svd(W,'econ');
        s     = diag(Sig);
        s_shr = max(s - 1/mu, 0);
        r     = nnz(s_shr > 0);
        if r > 0
            L = U(:,1:r) * diag(s_shr(1:r)) * V(:,1:r)';
        else
            L = zeros(m,n);
        end

        % --- sparse update (soft threshold) ---
        T = M - L + (1/mu)*Y;
        S = sign(T) .* max(abs(T) - lambda/mu, 0);

        % --- dual update & diagnostics ---
        Z      = M - L - S;
        Y      = Y + mu*Z;
        relres = norm(Z,'fro') / normM;
        obj    = sum(s_shr) + lambda*sum(abs(S(:)));

        if verbose && (mod(k,10)==0 || k==1)
            fprintf('%5d  %10.3e  %10.4e  %8d  %8d  %8.2e\n',...
                    k, relres, obj, r, nnz(S), mu);
        end

        mu = min(mu*mu_factor, mu_max);
    end

    if verbose
        fprintf('Done in %d iters, relres=%.3e, rank(L)=%d, nnz(S)=%d\n',...
                k, relres, r, nnz(S));
    end

    out = struct('iter',k,'relres',relres,'obj',obj,...
                 'rankL',r,'nnzS',nnz(S));
end

function v = get_opt(opts, field, defaultv)
    if isfield(opts,field) && ~isempty(opts.(field))
        v = opts.(field);
    else
        v = defaultv;
    end
end
