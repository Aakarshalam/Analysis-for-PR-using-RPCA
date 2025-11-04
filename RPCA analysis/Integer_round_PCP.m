%% RPCA from continuous sampling with integer-projected (round) PCP
% End-to-end script: builds a high-rate multitone, folds it, samples both at a
% lower rate, and recovers the unfolded samples via Hankel(Δy) + RPCA with
% integer projection (ROUND) on the sparse term in the scaled domain.

clear; close all; clc;

%% ---------------- USER PARAMETERS ----------------
% Multi-tone signal
component_freqs = [300, 1200, 3000];   % Hz
component_amps  = [1.00, 0.60, 0.40];
component_phs   = [0.00, 0.3*pi, -0.2*pi];

FS_CONT  = 200e3;       % "continuous" high-rate (Hz)
DURATION = 0.02;        % seconds

lambda = 1.2;           % folding half-range ([-lambda, +lambda))
alpha  = 2*lambda;        % scaling amplitude used in scaled-domain PCP

% Actual sampling rate (the discrete-time sequence used by RPCA)
FS_SAMPLE = 10e3;        % Hz

% time zoom for plots (sampled)
zoom_t0 = 0.002; zoom_t1 = 0.006;

% RPCA/IALM options
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%% -------- 1) CONSTRUCT CONTINUOUS SIGNALS (high-rate) ------
t_cont = 0:1/FS_CONT:DURATION-1/FS_CONT;

x_cont = zeros(size(t_cont));
for k = 1:numel(component_freqs)
    x_cont = x_cont + component_amps(k) * ...
        sin(2*pi*component_freqs(k)*t_cont + component_phs(k));
end

% Apply amplitude folding (nonlinear) at continuous grid
x_cont_fold = fold_centered(x_cont, lambda);

%% -------- 2) SAMPLE BOTH SIGNALS at FS_SAMPLE (from continuous) ---------
t_samp = 0:1/FS_SAMPLE:DURATION-1/FS_SAMPLE;
idx_samp = round(t_samp*FS_CONT) + 1;                 % pick exact samples from high-rate grid
idx_samp = min(max(idx_samp,1), numel(t_cont));       % guard

x_samp      = x_cont(idx_samp);       % true (unfolded) samples at FS_SAMPLE
y_samp_fold = x_cont_fold(idx_samp);  % folded samples at FS_SAMPLE (observation for RPCA)

%% -------- 3) FOLDING VISUALS (high-rate + sampled zoom) -----
figure('Name','Time domain (high-rate)','NumberTitle','off','Position',[60 60 1100 320]);
plot(t_cont, x_cont, 'LineWidth', 1.0); hold on;
plot(t_cont, x_cont_fold, '--', 'LineWidth', 1.0);
grid on; xlim([0, DURATION]);
xlabel('Time (s)'); ylabel('Amplitude');
legend('original (high-rate)','folded (high-rate)');
title(sprintf('High-rate signals (\\lambda = %.3g)', lambda));

% Zoom window (sampled)
zi = (t_samp >= zoom_t0) & (t_samp <= zoom_t1);
figure('Name','Time domain (sampled, zoom)','NumberTitle','off','Position',[60 400 1100 320]);
stem(t_samp(zi), x_samp(zi), 'filled'); hold on;
stem(t_samp(zi), y_samp_fold(zi), 'filled','MarkerFaceColor',[0.9 0.2 0.2]);
grid on; xlabel('Time (s)'); ylabel('Amplitude');
legend('original sampled','folded sampled');
title(sprintf('Sampled signals (zoom %.3f–%.3f s)', zoom_t0, zoom_t1));

%% -------- 4) RPCA on Hankel(Δ y) with INTEGER (round) PROJECTION --------
% Work on first difference of the folded samples, build a Hankel matrix,
% solve PCP in the scaled domain (divide by alpha), project sparse to integers,
% then unscale and reconstruct x at the sample instants.

% Difference order
dord = 1;                               % use Δ^1
Dy   = apply_diff(y_samp_fold, dord);   % size m = N-1
m    = numel(Dy);

% Choose a Hankel height L (ensure feasibility)
r_est = estimate_rank_from_dx(Dy);      % quick rank guess
[L_list, feasible] = pick_L_list(m, r_est);
if ~feasible
    error('Not enough samples: m=%d, r_est=%d. Increase FS_SAMPLE or DURATION.', m, r_est);
end
L = L_list(round(numel(L_list)/2));     % near-balanced feasible choice

% Build Hankel
H  = hankel_from_vector(Dy, L);

% Normalize & scale to the "bar" domain
scaleH = median(abs(H(:)) + eps);
Hn     = H / scaleH;
Mbar   = Hn / alpha;

% Run integer-projected PCP (round)
lam_pcp   = 1/sqrt(max(size(Mbar)));
proj_mode = 'round';
[HLbar, ~, out] = rpca_pcp_integer(Mbar, lam_pcp, opts, proj_mode);

% Unscale back to original domain
HLn = alpha * HLbar;
HL  = HLn * scaleH;

% De-Hankelize to recover Δ^1 x_hat, integrate back to x_hat (up to constant),
% then align to the proper modulo branch via integer k*
D_hat = dehankel_to_vector(HL);
x_hat = integrate_from_diffs(y_samp_fold, D_hat, dord);
kstar = median( round( (x_hat - y_samp_fold)/(2*lambda) ) );
x_hat = x_hat - 2*lambda*kstar;

% Quality vs ground-truth samples (x_samp)
rel_err = norm(x_samp - x_hat)/max(norm(x_samp),eps);
snr_db  = 20*log10( max(norm(x_samp),eps)/max(norm(x_samp - x_hat),eps) );
fprintf('[BASE] RPCA int-proj(round): L=%d | %dx%d | relres=%.2e | RelErr=%.3g | SNR=%.2f dB\n', ...
        L, size(H,1), size(H,2), out.relres, rel_err, snr_db);

% Plot recovery at sample instants
figure('Name','RPCA recovery at FS\_SAMPLE','NumberTitle','off','Position',[60 760 1100 320]);
plot(t_samp, x_samp, 'LineWidth', 1.25); hold on;
plot(t_samp, x_hat, '--', 'LineWidth', 1.25); grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Recovery (int-proj, round) at f_s=%.0f Hz | L=%d | SNR=%.2f dB', FS_SAMPLE, L, snr_db));
legend('x (true sampled)','\hat{x} (recovered)','Location','best');

disp('Done. Adjust lambda, FS_SAMPLE, and component_freqs to observe folding/recovery behavior.');

%% ====================== Local helpers (keep below main script) ======================

function y = fold_centered(x, lambda)
% Amplitude folding into [-lambda, +lambda)
% Equivalent to modulo on amplitude with symmetric wraparound.
    y = mod(x + lambda, 2*lambda) - lambda;
end

function D = apply_diff(x, order)
    if nargin<2 || isempty(order), order = 1; end
    if order==1
        D = diff(x);
    elseif order==2
        D = diff(diff(x));
    else
        error('Only order 1 or 2 supported.');
    end
end

function x_hat = integrate_from_diffs(y, D_hat, order)
% Invert Δ or Δ² using the first sample of y as boundary condition.
    if order==1
        x_hat = [y(1), y(1) + cumsum(D_hat)];
    else
        d1 = [D_hat(1), cumsum(D_hat)];
        x_hat = [y(1), y(1) + cumsum(d1)];
    end
end

function H = hankel_from_vector(v, L)
    N = numel(v); M = N - L + 1;
    if M <= 0, error('Hankel height L too large for length %d.', N); end
    H = hankel(v(1:L), v(L:N));
end

function v = dehankel_to_vector(H)
% Anti-diagonal averaging back to a 1-D sequence
    [L,M] = size(H); Lm = L + M - 1;
    v = zeros(1, Lm); c = zeros(1, Lm);
    for i=1:L
        for j=1:M
            k=i+j-1; v(k)=v(k)+H(i,j); c(k)=c(k)+1;
        end
    end
    v = v ./ c;
end

% ---------- Integer-projected RPCA (PCP via IALM in scaled domain) ----------
function [Lbar,Sbar,out] = rpca_pcp_integer(Mbar, lambda_pcp, opts, proj_mode)
% Solve in scaled domain:
%   min ||Lbar||_* + lambda*||Sbar||_1
%   s.t. Mbar = Lbar + Sbar,   Sbar ∈ Z^{m×n}
% We do standard IALM, but after the soft-threshold step on Sbar we PROJECT
% to INTEGERS using 'round' (default), which is symmetric for +/- folds.

    if ~isfloat(Mbar), Mbar = double(Mbar); end
    [m,n] = size(Mbar);

    if ~exist('lambda_pcp','var') || isempty(lambda_pcp)
        lambda_pcp = 1/sqrt(max(m,n));
    end
    if ~exist('opts','var') || isempty(opts), opts = struct(); end
    if ~exist('proj_mode','var') || isempty(proj_mode), proj_mode = 'round'; end

    tol       = get_opt(opts,'tol',1e-7);
    max_iter  = get_opt(opts,'max_iter',1000);
    verbose   = get_opt(opts,'verbose',1);
    mu        = get_opt(opts,'mu',[]);
    mu_factor = get_opt(opts,'mu_factor',1.5);
    mu_max    = get_opt(opts,'mu_max',1e7);

    normM = norm(Mbar,'fro');
    if normM==0
        Lbar=zeros(m,n); Sbar=zeros(m,n);
        out=struct('iter',0,'relres',0,'obj',0,'rankL',0,'nnzS',0);
        return;
    end

    if isempty(mu)
        try smax = svds(Mbar,1); catch, smax = norm(Mbar,2); end
        mu = 1.25 / max(smax, eps);
    end

    Lbar = zeros(m,n);
    Sbar = zeros(m,n);
    Y    = Mbar / max(norm(Mbar,2), eps);

    relres = inf; k = 0; r = 0; obj = NaN;

    if verbose
        fprintf('RPCA-PCP-INT (round): lambda=%.3g, tol=%.1e\n', lambda_pcp, tol);
        fprintf('%5s  %10s  %10s  %8s  %8s  %8s\n','iter','relres','obj','rank(L)','nnz(S)','mu');
    end

    while (relres > tol) && (k < max_iter)
        k = k + 1;

        % === Lbar update: SVT ===
        W = Mbar - Sbar + (1/mu)*Y;
        [U,Sig,V] = svd(W,'econ');
        s = diag(Sig);
        s_shr = max(s - 1/mu, 0);
        r = nnz(s_shr > 0);
        if r>0
            Lbar = U(:,1:r) * diag(s_shr(1:r)) * V(:,1:r)';
        else
            Lbar = zeros(m,n);
        end

        % === Sbar update: soft-threshold then integer projection ===
        T = Mbar - Lbar + (1/mu)*Y;
        Sbar_cont = sign(T) .* max(abs(T) - lambda_pcp/mu, 0);

        switch lower(proj_mode)
            case 'round'
                Sbar = round(Sbar_cont);
            case 'ceil'
                Sbar = ceil(Sbar_cont);
            otherwise
                error('Unknown proj_mode: %s (use "round" or "ceil")', proj_mode);
        end

        % === Dual & bookkeeping ===
        Z = Mbar - Lbar - Sbar;
        Y = Y + mu * Z;

        relres = norm(Z,'fro') / normM;
        obj = sum(s_shr) + lambda_pcp * sum(abs(Sbar(:))); % value after projection

        if verbose && (mod(k,10)==0 || k==1)
            fprintf('%5d  %10.3e  %10.4e  %8d  %8d  %8.2e\n', ...
                k, relres, obj, r, nnz(Sbar), mu);
        end

        mu = min(mu * mu_factor, mu_max);
    end

    if verbose
        fprintf('Done in %d iters, relres=%.3e, rank(L)=%d, nnz(S)=%d\n',k,relres,r,nnz(Sbar));
    end

    out = struct('iter',k,'relres',relres,'obj',obj,'rankL',r,'nnzS',nnz(Sbar));
end

function v = get_opt(opts, field, defaultv)
    if isfield(opts,field) && ~isempty(opts.(field)), v = opts.(field); else, v = defaultv; end
end

% ---- feasible Hankel heights for m = length(Δ^d y) ----
function [L_list, feasible] = pick_L_list(m, r_est)
    if isempty(r_est), r_est = 2; end
    Lmin = max(2, r_est+1);
    Lmax = m - Lmin + 1;
    feasible = (Lmax >= Lmin);
    if ~feasible, L_list = []; return; end
    cand = unique(round([0.3 0.4 0.5 0.6 0.7]*m));
    cand = cand(cand >= Lmin & cand <= Lmax);
    if isempty(cand), cand = floor((Lmin + Lmax)/2); end
    L_list = unique([Lmin, cand, Lmax]);
end

% ---- rank estimate from Hankel of Δx (or Δ²x) ----
function r_est = estimate_rank_from_dx(Dx)
    m = numel(Dx); if m < 8, r_est = 2; return; end
    L = max(4, floor(0.5*m)); H = hankel_from_vector(Dx, L);
    s = svd(H,'econ'); s = s / (s(1)+eps);
    cs = cumsum(s.^2)/sum(s.^2);
    r_est = find(cs>=0.999,1,'first');
    r_est = min(max(r_est,2), 30);
end
