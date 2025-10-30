function demo_unlimited_rpca_fixedT_multitone_discreteLam()
% RPCA recovery under modulo folding with fixed duration T.
% Each sample uses a random lambda[n] ∈ {2,3,4} with equal probability.

rng(0);

%% -------- 0) User params (edit here) --------
T        = 0.15;        % total duration [s] (fixed)
fs0      = 7000;        % baseline sampling rate [Hz]
lam_support = [2 3 4];  % per-sample discrete λ choices (equiprobable)

% ===== Choose signal =====
% A) explicit multitone (YOU set freqs/amps/phis)
signal.type  = 'multitone';
signal.freqs = [120 225 330];        % Hz
signal.amps  = [1.0  2.0  2.0];      % amplitudes
signal.phis  = pi*[0    0    0];     % radians

% B) random multitone (uncomment to use)
% signal.type     = 'multitone_random';
% signal.K        = 6;               % number of tones
% signal.f_range  = [50 350];        % Hz (upper bound clipped at 0.49*fs)
% signal.a_range  = [0.4 1.5];       % amplitude range
% signal.phi_mode = 'uniform';       % 'uniform' or 'zero'
% signal.min_df   = 10;              % optional min spacing (Hz)

% RPCA options
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%% -------- 1) Baseline at fs0 --------
N0 = max(8, round(T*fs0));
t0 = (0:N0-1)/fs0;

[x0, meta0] = make_signal(t0, signal, fs0);
dord = meta0.diff_order; if isempty(dord), dord = 1; end

% Per-sample λ[n] ∈ {2,3,4} and folding
lam0 = draw_lambda_discrete(N0, lam_support);
y0   = fold_centered_var(x0, lam0);

% Show original vs folded
figure('Name','Original vs Folded (fixed T, discrete lambda)','Color','w');
subplot(2,1,1); plot(t0, x0, 'LineWidth',1.1); grid on;
title(sprintf('Original x(t) | %s | T=%.4fs, fs=%.0fHz, N=%d', ...
      upper(signal.type), T, fs0, N0));
xlabel('t [s]'); ylabel('amplitude');

subplot(2,1,2); plot(t0, y0, 'LineWidth',1.1); grid on;
title('\lambda[n] \in \{2,3,4\} (equiprobable)'); xlabel('t [s]'); ylabel('amplitude');

% Optional: histogram of lambda choices
figure('Name','Lambda histogram (baseline)','Color','w');
histogram(lam0,'BinMethod','integers'); grid on;
xlabel('\lambda value'); ylabel('count'); title('\lambda[n] ∈ {2,3,4}');

% Wrap density diag
ktrue0 = round( (x0 - y0) ./ (2*lam0) );
wraps0 = find(diff(ktrue0)~=0);
fprintf('[BASE] wraps=%d  (%.2f%% of samples)\n', numel(wraps0), 100*numel(wraps0)/N0);

%% -------- 2) Hankel(Δ^d y0) + RPCA at fs0 (L sweep) --------
D_y0  = apply_diff(y0, dord);
m0    = numel(D_y0);
r_est = meta0.r_est; if isempty(r_est), r_est = estimate_rank_from_dx(D_y0); end
[L_list0, L_feasible] = pick_L_list(m0, r_est);
if ~L_feasible
    warning('Not enough samples at baseline: m=%d, r_est=%d. Skipping RPCA.', m0, r_est);
else
    snr_list = nan(size(L_list0));
    best = struct('snr', -inf);

    fprintf('\n--- RPCA on Hankel(Δ^%d y) at fs0: sweeping feasible L ---\n', dord);
    for ii=1:numel(L_list0)
        L = L_list0(ii);
        H = hankel_from_vector(D_y0, L);

        % Normalize to stabilize PCP
        scaleH = median(abs(H(:)) + eps);
        Hn = H / scaleH;

        % PCP
        lam_pcp = 1/sqrt(max(size(Hn)));
        [HLn, ~, out] = rpca_pcp(Hn, lam_pcp, opts);
        HL = HLn * scaleH;

        % Back to signal: de-Hankelize, integrate, then elementwise 2λ[n] alignment
        D_hat = dehankel_to_vector(HL);
        x_hat = integrate_from_diffs(y0, D_hat, dord);

        kvec  = round( (x_hat - y0) ./ (2*lam0) );  % integer sequence
        x_hat = x_hat - 2*lam0 .* kvec;             % elementwise adjust

        % Quality
        snr_list(ii) = 20*log10( max(norm(x0),eps) / max(norm(x0 - x_hat),eps) );
        fprintf('  L=%3d | %dx%d | relres=%.2e | SNR=%.2f dB\n', ...
                L, size(H,1), size(H,2), out.relres, snr_list(ii));

        if snr_list(ii) > best.snr
            best = struct('L',L,'x_hat',x_hat,'D_hat',D_hat,'HL',HL,'snr',snr_list(ii));
        end
    end

    % Plots
    figure('Name','Recovery at fs0 (best L)','Color','w');
    plot(t0, x0, 'LineWidth',1.25); hold on;
    plot(t0, best.x_hat, '--', 'LineWidth',1.25); grid on;
    xlabel('t [s]'); ylabel('amplitude');
    title(sprintf('Best recovery at fs=%.0f Hz | L=%d | SNR=%.2f dB', fs0, best.L, best.snr));
    legend('x','\hat{x}','Location','best');

    if numel(L_list0)>1
        figure('Name','Effect of Hankel height (fs0)','Color','w');
        plot(L_list0, snr_list, 'o-','LineWidth',1.25); grid on;
        xlabel('Hankel height L'); ylabel('SNR (dB)');
        title(sprintf('Recovery SNR vs L (Δ^%d, discrete \\lambda[n])', dord));
    end
end

%% -------- 3) Sweep fs with fixed T (N varies; fresh discrete λ[n] each fs) --------
fs_min = 240; fs_max = 2e4; npts = 200;
fs_vec = round(logspace(log10(fs_min), log10(fs_max), npts));

err_rel = nan(size(fs_vec));
snr_db  = nan(size(fs_vec));
wrap_pct = nan(size(fs_vec));

fprintf('\n--- Sweep fs with fixed T=%.4fs (discrete λ[n], tiny-N enabled) ---\n', T);
for kk=1:numel(fs_vec)
    fs_k = fs_vec(kk);
    N_k  = max(8, round(T*fs_k));
    t_k  = (0:N_k-1)/fs_k;

    % signal at fs_k
    [x_k, meta_k] = make_signal(t_k, signal, fs_k);
    dord_k = meta_k.diff_order; if isempty(dord_k), dord_k = 1; end

    % per-sample λ and folding
    lam_k = draw_lambda_discrete(N_k, lam_support);
    y_k   = fold_centered_var(x_k, lam_k);

    % wrap density
    ktrue_k = round( (x_k - y_k) ./ (2*lam_k) );
    wraps_k = find(diff(ktrue_k)~=0);
    wrap_pct(kk) = 100*numel(wraps_k)/N_k;

    % Hankel(Δ^d y) with feasible L
    D_yk = apply_diff(y_k, dord_k);
    m_k  = numel(D_yk);

    r_est_k = meta_k.r_est; if isempty(r_est_k), r_est_k = estimate_rank_from_dx(D_yk); end
    [L_listk, feasible] = pick_L_list(m_k, r_est_k);
    if ~feasible, continue; end

    L_k = L_listk(round(numel(L_listk)/2)); % near-balanced feasible
    H_k = hankel_from_vector(D_yk, L_k);
    scaleH = median(abs(H_k(:)) + eps);
    Hn_k = H_k / scaleH;

    lam_pcp = 1/sqrt(max(size(Hn_k)));
    [HLn_k, ~, outk] = rpca_pcp(Hn_k, lam_pcp, opts); %#ok<NASGU>
    HL_k = HLn_k * scaleH;

    D_hat_k = dehankel_to_vector(HL_k);
    x_hat_k = integrate_from_diffs(y_k, D_hat_k, dord_k);

    % elementwise alignment with λ[n]
    kvec_k  = round( (x_hat_k - y_k) ./ (2*lam_k) );
    x_hat_k = x_hat_k - 2*lam_k .* kvec_k;

    % metrics
    err_rel(kk) = norm(x_k - x_hat_k) / max(norm(x_k), eps);
    snr_db(kk)  = 20*log10( max(norm(x_k),eps) / max(norm(x_k - x_hat_k),eps) );
end

% Plots
figure('Name','Error vs Sampling (fixed T, discrete λ[n])','Color','w');
subplot(1,2,1);
semilogx(fs_vec, err_rel, 'o-','LineWidth',1.25); grid on; hold on;
yyaxis right; semilogx(fs_vec, wrap_pct, 's--','LineWidth',1.1);
ylabel('Wrap density (%)');
yyaxis left; xlabel('Sampling frequency f_s (Hz)');
ylabel('Relative error  ||x-\hat{x}||/||x||');
title(sprintf('Fixed T=%.4fs: error & wrap density vs f_s (discrete \\lambda[n])', T));
legend('Rel. error','Wrap density','Location','best');

subplot(1,2,2);
semilogx(fs_vec, snr_db, 'o-','LineWidth',1.25); grid on;
xlabel('Sampling frequency f_s (Hz)');
ylabel('SNR (dB)');
title('Fixed T: recovery SNR vs f_s');

%% -------- Report smallest fs with error < 0.05% ----------
target_err = 5e-1;   % 0.05%
valid = ~isnan(err_rel);
candidates = find(valid & err_rel <= target_err);

if isempty(candidates)
    fprintf('[TARGET] No fs in the sweep met rel. error < %.4g (discrete λ[n]).\n', target_err);
else
    [~, idx_in_candidates_sorted] = min(fs_vec(candidates));
    idx_star = candidates(idx_in_candidates_sorted);
    fs_star   = fs_vec(idx_star);
    err_star  = err_rel(idx_star);
    snr_star  = snr_db(idx_star);
    wrap_star = wrap_pct(idx_star);
    N_star    = round(T * fs_star);
    fprintf('[TARGET] rel. error %.4g at fs = %.0f Hz (N = %d)\n', err_star, fs_star, N_star);
    fprintf('         Wrap density: %.2f%%   |   SNR: %.2f dB   |   discrete λ[n]\n', ...
            wrap_star, snr_star);
end

end

%% ================= helpers =================
function y = fold_centered_var(x, lambda_vec)
% Centered modulo with per-sample lambda: y[n] = mod(x[n]+λ[n], 2λ[n]) - λ[n]
    y = mod(x + lambda_vec, 2.*lambda_vec) - lambda_vec;
end

function lam = draw_lambda_discrete(N, support)
% Draw N i.i.d. samples uniformly from the finite set "support".
% Example: support = [2 3 4] -> each with probability 1/3.
    idx = randi(numel(support), 1, N);
    lam = support(idx);
end

function D = apply_diff(x, order)
    if isempty(order), order = 1; end
    if order==1
        D = diff(x);
    elseif order==2
        D = diff(diff(x));
    else
        error('Only order 1 or 2 supported.');
    end
end

function x_hat = integrate_from_diffs(y, D_hat, order)
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
    [L,M] = size(H); Lm = L + M - 1;
    v = zeros(1, Lm); c = zeros(1, Lm);
    for i=1:L
        for j=1:M
            k=i+j-1; v(k)=v(k)+H(i,j); c(k)=c(k)+1;
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
    tol = get_opt(opts,'tol',1e-7); max_iter = get_opt(opts,'max_iter',1000);
    verbose = get_opt(opts,'verbose',1); mu = get_opt(opts,'mu',[]);
    mu_factor = get_opt(opts,'mu_factor',1.5); mu_max = get_opt(opts,'mu_max',1e7);

    normM = norm(M,'fro'); 
    if normM==0, L=zeros(m,n); S=zeros(m,n);
        out=struct('iter',0,'relres',0,'obj',0,'rankL',0,'nnzS',0); return; end
    if isempty(mu)
        try smax = svds(M,1); catch, smax = norm(M,2); end
        mu = 1.25 / max(smax, eps);
    end

    L=zeros(m,n); S=zeros(m,n); Y = M / max(norm(M,2), eps);
    relres=inf; k=0; r=0; obj=NaN;
    if verbose
        fprintf('RPCA-PCP: lambda=%.3g, tol=%.1e\n', lambda, tol);
        fprintf('%5s  %10s  %10s  %8s  %8s  %8s\n','iter','relres','obj','rank(L)','nnz(S)','mu');
    end

    while (relres>tol) && (k<max_iter)
        k=k+1;
        W = M - S + (1/mu)*Y;
        [U,Sig,V]=svd(W,'econ');
        s = diag(Sig); s_shr = max(s - 1/mu, 0); r = nnz(s_shr>0);
        if r>0, L = U(:,1:r)*diag(s_shr(1:r))*V(:,1:r)'; else, L=zeros(m,n); end

        T = M - L + (1/mu)*Y;
        S = sign(T).*max(abs(T) - lambda/mu, 0);

        Z = M - L - S; Y = Y + mu*Z;
        relres = norm(Z,'fro')/normM;
        obj = sum(s_shr) + lambda*sum(abs(S(:)));
        if verbose && (mod(k,10)==0 || k==1)
            fprintf('%5d  %10.3e  %10.4e  %8d  %8d  %8.2e\n',k,relres,obj,r,nnz(S),mu);
        end
        mu = min(mu*mu_factor, mu_max);
    end
    if verbose
        fprintf('Done in %d iters, relres=%.3e, rank(L)=%d, nnz(S)=%d\n',k,relres,r,nnz(S));
    end
    out = struct('iter',k,'relres',relres,'obj',obj,'rankL',r,'nnzS',nnz(S));
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

% ---- signal generator (explicit & random multitone) ----
function [x, meta] = make_signal(t, P, fs)
    if nargin<3, fs = []; end
    meta = struct('diff_order',1,'r_est',[],'K_eff',[]);
    switch lower(P.type)
        case 'multitone'   % YOU provide freqs/amps/phis
            assert(isfield(P,'freqs')&&isfield(P,'amps')&&isfield(P,'phis'), ...
                'For signal.type="multitone", set fields: freqs, amps, phis (rad).');
            K = numel(P.freqs);
            assert(numel(P.amps)==K && numel(P.phis)==K, 'freqs/amps/phis must match in length.');
            if ~isempty(fs) && any(P.freqs >= fs/2)
                warning('Some freqs exceed Nyquist (fs/2). Consider lowering them or raising fs.');
            end
            x = zeros(size(t));
            for k=1:K
                x = x + P.amps(k)*cos(2*pi*P.freqs(k)*t + P.phis(k));
            end
            meta.K_eff = K; meta.r_est = 2*K; meta.diff_order = 1;

        case 'multitone_random'
            assert(isfield(P,'K')&&isfield(P,'f_range')&&isfield(P,'a_range'), ...
                'Need fields: K, f_range [Hz], a_range.');
            K = P.K;
            fr = P.f_range; ar = P.a_range;
            if fr(1) <= 0, fr(1)=1; end
            if ~isempty(fs), fr(2) = min(fr(2), 0.49*fs); end
            f = sort(fr(1) + (fr(2)-fr(1))*rand(1,K));
            if isfield(P,'min_df') && P.min_df>0
                for i=2:K
                    if f(i)-f(i-1) < P.min_df
                        f(i) = f(i-1)+P.min_df;
                    end
                end
                if ~isempty(fs), f(f>=0.49*fs) = 0.49*fs - 1; end
            end
            a  = ar(1) + (ar(2)-ar(1))*rand(1,K);
            ph = 2*pi*rand(1,K);
            if isfield(P,'phi_mode') && strcmpi(P.phi_mode,'zero'), ph = zeros(1,K); end
            x = zeros(size(t));
            for k=1:K, x = x + a(k)*cos(2*pi*f(k)*t + ph(k)); end
            meta.K_eff = K; meta.r_est = 2*K; meta.diff_order = 1;

        otherwise
            error('Unknown signal.type: %s', P.type);
    end
end
