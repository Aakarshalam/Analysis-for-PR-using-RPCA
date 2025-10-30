function demo_unlimited_rpca_fixedT_multitone()
% RPCA recovery under modulo folding with fixed duration T.
rng(0);

%% -------- 0) User params (edit here) --------
T        = 0.15;        % total duration [s] (fixed)
fs0      = 240;        % baseline sampling rate [Hz]
lambda   = 7.0;         % folding threshold (range [-lambda,lambda])

% ===== Pick exactly one =====
% Option A: explicit multitone (YOU set freqs/amps/phis)
signal.type  = 'multitone';
signal.freqs = [120 225 330];           % Hz  (edit me)
signal.amps  = [2.0  2.5 1.8];         %     (edit me)
signal.phis  = pi*[0 0 0];       % rad (edit me)

% Option B: random multitone (uncomment to use)
% signal.type     = 'multitone_random';
% signal.K        = 6;               % number of tones
% signal.f_range  = [50 350];        % Hz
% signal.a_range  = [0.4 1.5];       % amplitudes
% signal.phi_mode = 'uniform';       % 'uniform' or 'zero'
% signal.min_df   = 10;              % min spacing in Hz (optional)

% For other built-ins, you can still choose:
% 'mono','dual','multitone5','multitone7','am','fm','chirp','damped',
% 'trend_plus_tone','piecewise_constant','square_bandlimited','spikes_plus_tone'
% (Those branches remain available; see make_signal)

% Common tone params (used by some built-ins)
signal.A1=2.0; signal.f1=120; signal.phi1=0.30*pi;
signal.A2=1.2; signal.f2=300; signal.phi2=-0.55*pi;

% Extra params (used by certain types)
signal.am_m   = 0.6;   signal.am_fm = 20;
signal.fm_beta= 0.7;   signal.fm_fm = 18;
signal.chirp_f0 = 60;  signal.chirp_B = 240;
signal.damp_rho = 0.997;
signal.trend_c1 = 0.5; signal.trend_c2 = 0.0;
signal.square_f0 = 60; signal.square_K = 9;
signal.spike_ct = 10;

% RPCA/IALM options
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%% -------- 1) Baseline at fs0 --------
N0 = max(8, round(T*fs0));
t0 = (0:N0-1)/fs0;

[x0, meta0] = make_signal(t0, signal, fs0);
dord = meta0.diff_order; if isempty(dord), dord = 1; end

% Fold
y0 = fold_centered(x0, lambda);

% Show original vs folded
figure('Name','Original vs Folded (fixed T)','Color','w');
subplot(2,1,1); plot(t0, x0, 'LineWidth',1.1); grid on;
title(sprintf('Original x(t) | %s | T=%.4fs, fs=%.0fHz, N=%d', ...
      upper(signal.type), T, fs0, N0));
xlabel('t [s]'); ylabel('amplitude');
subplot(2,1,2); plot(t0, y0, 'LineWidth',1.1); grid on;
title(sprintf('Folded y(t) in [ -\\lambda, \\lambda ], \\lambda=%.2f', lambda));
xlabel('t [s]'); ylabel('amplitude');

% Wrap density diag
ktrue0 = round((x0 - y0)/(2*lambda));
wraps0 = find(diff(ktrue0)~=0);
fprintf('[BASE] wraps=%d  (%.2f%% of samples)\n', numel(wraps0), 100*numel(wraps0)/N0);

%% -------- 2) Hankel on Δ^d y + RPCA at fs0 (L sweep) --------
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

        % Back to signal
        D_hat = dehankel_to_vector(HL);
        x_hat = integrate_from_diffs(y0, D_hat, dord);
        kstar = median( round( (x_hat - y0)/(2*lambda) ) );
        x_hat = x_hat - 2*lambda*kstar;

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
        title(sprintf('Recovery SNR vs L (Δ^%d, fixed T, baseline fs)', dord));
    end
end

%% -------- 3) Sweep fs with fixed T (N varies; tiny-N OK) --------
fs_min = 240; fs_max = 2e4; npts = 200;
fs_vec = round(logspace(log10(fs_min), log10(fs_max), npts));

err_rel = nan(size(fs_vec));
snr_db  = nan(size(fs_vec));
wrap_pct = nan(size(fs_vec));

fprintf('\n--- Sweep fs with fixed T=%.4fs (tiny-N enabled) ---\n', T);
for kk=1:numel(fs_vec)
    fs_k = fs_vec(kk);
    N_k  = max(8, round(T*fs_k));
    t_k  = (0:N_k-1)/fs_k;

    % signal at fs_k
    [x_k, meta_k] = make_signal(t_k, signal, fs_k);
    dord_k = meta_k.diff_order; if isempty(dord_k), dord_k = 1; end

    % fold
    y_k = fold_centered(x_k, lambda);

    % wrap density
    ktrue_k = round((x_k - y_k)/(2*lambda));
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
    [HLn_k, ~, outk] = rpca_pcp(Hn_k, lam_pcp, opts); 
    HL_k = HLn_k * scaleH;

    D_hat_k = dehankel_to_vector(HL_k);
    x_hat_k = integrate_from_diffs(y_k, D_hat_k, dord_k);
    kstar_k = median( round( (x_hat_k - y_k)/(2*lambda) ) );
    x_hat_k = x_hat_k - 2*lambda*kstar_k;

    err_rel(kk) = norm(x_k - x_hat_k) / max(norm(x_k), eps);
    snr_db(kk)  = 20*log10( max(norm(x_k),eps) / max(norm(x_k - x_hat_k),eps) );
end

% Plots
figure('Name','Error vs Sampling (fixed T, tiny-N OK)','Color','w');
subplot(1,2,1);
semilogx(fs_vec, err_rel, 'o-','LineWidth',1.25); grid on; hold on;
yyaxis right; semilogx(fs_vec, wrap_pct, 's--','LineWidth',1.1);
ylabel('Wrap density (%)');
yyaxis left; xlabel('Sampling frequency f_s (Hz)');
ylabel('Relative error  ||x-\hat{x}||/||x||');
title(sprintf('Fixed T=%.4fs: error & wrap density vs f_s (%s)', T, signal.type));
legend('Rel. error','Wrap density','Location','best');

subplot(1,2,2);
semilogx(fs_vec, snr_db, 'o-','LineWidth',1.25); grid on;
xlabel('Sampling frequency f_s (Hz)');
ylabel('SNR (dB)');
title('Fixed T: recovery SNR vs f_s');

%% -------- Find fs where error < 0.05% and report wrap density ----------
target_err = 5e-4;   % 0.05%
valid = ~isnan(err_rel);
candidates = find(valid & err_rel <= target_err);

if isempty(candidates)
    fprintf('[TARGET] No fs in the current sweep met rel. error < %.4g.\n', target_err);
else
    [~, idx_in_candidates_sorted] = min(fs_vec(candidates));
    idx_star = candidates(idx_in_candidates_sorted);
    fs_star   = fs_vec(idx_star);
    err_star  = err_rel(idx_star);
    snr_star  = snr_db(idx_star);
    wrap_star = wrap_pct(idx_star);
    N_star    = round(T * fs_star);
    fprintf('[TARGET] rel. error %.4g at fs = %.0f Hz (N = %d)\n', err_star, fs_star, N_star);
    fprintf('         Wrap density: %.2f%%   |   SNR: %.2f dB   |   type: %s\n', ...
            wrap_star, snr_star, signal.type);
end


%% ================= helpers =================
function y = fold_centered(x, lambda)
    y = mod(x + lambda, 2*lambda) - lambda;
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

% ---- signal generator (now supports explicit & random multitone) ----
function [x, meta] = make_signal(t, P, fs)
    if nargin<3, fs = []; end
    meta = struct('diff_order',1,'r_est',[],'K_eff',[]);
    switch lower(P.type)
        case 'multitone'   % <<< YOU provide freqs/amps/phis
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
                % greedy enforce min spacing
                for i=2:K
                    if f(i)-f(i-1) < P.min_df
                        f(i) = f(i-1)+P.min_df;
                    end
                end
                if ~isempty(fs)
                    f(f>=0.49*fs) = 0.49*fs - 1;
                end
            end
            a = ar(1) + (ar(2)-ar(1))*rand(1,K);
            if isfield(P,'phi_mode') && strcmpi(P.phi_mode,'zero')
                ph = zeros(1,K);
            else
                ph = 2*pi*rand(1,K);
            end
            x = zeros(size(t));
            for k=1:K
                x = x + a(k)*cos(2*pi*f(k)*t + ph(k));
            end
            meta.K_eff = K; meta.r_est = 2*K; meta.diff_order = 1;

        % --- other prebuilt cases remain available (same as before) ---
        case 'mono'
            x = P.A1*cos(2*pi*P.f1*t + P.phi1);
            meta.K_eff=1; meta.r_est=2;
        case 'dual'
            x = P.A1*cos(2*pi*P.f1*t + P.phi1) + P.A2*cos(2*pi*P.f2*t + P.phi2);
            meta.K_eff=2; meta.r_est=4;
        case 'multitone5'
            freqs = [60 90 120 180 300]; amps=[1.0 0.8 1.5 0.7 1.2]; phis=pi*[0 0.1 0.3 -0.2 0.5];
            x = zeros(size(t)); for k=1:numel(freqs), x = x + amps(k)*cos(2*pi*freqs(k)*t + phis(k)); end
            meta.K_eff=5; meta.r_est=10;
        case 'multitone7'
            freqs=[50 80 110 140 200 260 330]; amps=[0.9 0.7 1.2 0.8 1.0 0.6 0.5]; phis=pi*[0.2 0 0.3 -0.4 0.1 -0.2 0.5];
            x = zeros(size(t)); for k=1:numel(freqs), x = x + amps(k)*cos(2*pi*freqs(k)*t + phis(k)); end
            meta.K_eff=7; meta.r_est=14;
        case 'am'
            x = (1 + P.am_m*cos(2*pi*P.am_fm*t)) .* cos(2*pi*P.f1*t + P.phi1);
            meta.K_eff=3; meta.r_est=6;
        case 'fm'
            x = cos(2*pi*P.f1*t + P.fm_beta*sin(2*pi*P.fm_fm*t));
        case 'chirp'
            alpha = P.chirp_B / (t(end)-t(1)+eps);
            x = cos(2*pi*(P.chirp_f0*t + 0.5*alpha*t.^2));
        case 'damped'
            n = 0:numel(t)-1; x = P.A1*(P.damp_rho.^n).*cos(2*pi*P.f1*t + P.phi1);
            meta.K_eff=1; meta.r_est=2;
        case 'trend_plus_tone'
            trend = P.trend_c1*t + P.trend_c2*t.^2;
            x = trend + P.A1*cos(2*pi*P.f1*t + P.phi1);
            meta.diff_order = 2; % use Δ²
        case 'piecewise_constant'
            x = zeros(size(t));
            x(t>0.2*max(t) & t<=0.5*max(t)) = 0.8; x(t>0.7*max(t)) = 0.3;
            meta.K_eff=1; meta.r_est=2;
        case 'square_bandlimited'
            x = zeros(size(t)); f0=P.square_f0; K=P.square_K;
            for k=1:K, n=2*k-1; x = x + (4/pi)*(1/n)*sin(2*pi*n*f0*t); end
            meta.K_eff=K; meta.r_est=min(2*K,20);
        case 'spikes_plus_tone'
            x = 0.8*cos(2*pi*P.f1*t + P.phi1);
            idx = randperm(numel(t), min(P.spike_ct, numel(t)));
            x(idx) = x(idx) + 2.5*sign(randn(size(idx)));
            meta.K_eff=1; meta.r_est=2;
        otherwise
            error('Unknown signal.type: %s', P.type);
    end

    % If rank not set for complex models (FM/chirp), leave empty -> estimated later
    if any(strcmpi(P.type, {'fm','chirp'}))
        meta.r_est = [];
    end
end
