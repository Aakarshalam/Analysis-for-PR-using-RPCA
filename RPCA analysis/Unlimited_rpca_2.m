function demo_unlimited_rpca_fixedT_smallN()
% Fixed-time RPCA recovery under folding; supports very small N.
rng(0);

%% -------- 0) User params (edit here) --------
T      = 0.09;     % total duration [s] (fixed)
fs0    = 7000;       % baseline sampling rate [Hz]
lambda = 1;        % folding threshold ([-lambda, lambda])

% ----- choose a signal -----
A1=2.0; f1=120;  phi1=0.30*pi;
A2=1.2; f2=300;  phi2=-0.55*pi;
USE_TWO_TONE = true; % set false for mono tone

% estimated model order (Hankel rank <= 2K for K real tones)
K_tones = 2*(USE_TWO_TONE) + 1*(~USE_TWO_TONE);  % 2 or 1
r_est   = 2*K_tones;  % 4 (dual) or 2 (mono)

% RPCA/IALM options
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%% -------- 1) Baseline at fs0 (N0 = T*fs0) --------
N0 = max(8, round(T*fs0));      % tiny safety only; allow N0 < 40
t0 = (0:N0-1)/fs0;

if USE_TWO_TONE
    x0 = A1*cos(2*pi*f1*t0 ) + A2*cos(2*pi*f2*t0);
else
    x0 = A1*cos(2*pi*f1*t0 + phi1);
end

y0 = fold_centered(x0, lambda);

% Show original vs folded
figure('Name','Original vs Folded (fixed T)','Color','w');
subplot(2,1,1); plot(t0, x0, 'LineWidth',1.1); grid on;
title(sprintf('Original x(t) | T=%.4fs, fs=%.0fHz, N=%d', T, fs0, N0));
xlabel('t [s]'); ylabel('amplitude');
subplot(2,1,2); plot(t0, y0, 'LineWidth',1.1); grid on;
title(sprintf('Folded y(t) in [ -\\lambda, \\lambda ], \\lambda=%.2f', lambda));
xlabel('t [s]'); ylabel('amplitude');

% Wrap density diag
ktrue0 = round((x0 - y0)/(2*lambda));
wraps0 = find(diff(ktrue0)~=0);
fprintf('[BASE] wraps=%d  (%.2f%% of samples)\n', numel(wraps0), 100*numel(wraps0)/N0);

%% -------- 2) Hankel(Δy0) + RPCA at fs0 (L chosen feasibly) --------
dy0   = diff(y0);
m0    = numel(dy0);                      % Hankel length
[L_list0, L_feasible] = pick_L_list(m0, r_est);
if ~L_feasible
    warning('Not enough samples at baseline: m=%d, r_est=%d. Skipping RPCA.', m0, r_est);
else
    snr_list = nan(size(L_list0));
    best = struct('snr', -inf);

    fprintf('\n--- RPCA on Hankel(Δy) at fs0: sweeping feasible L ---\n');
    for ii=1:numel(L_list0)
        L = L_list0(ii);
        H = hankel_from_vector(dy0, L);

        % Normalize to stabilize PCP
        scaleH = median(abs(H(:)) + eps);
        Hn = H / scaleH;

        % PCP
        lam_pcp = 1/sqrt(max(size(Hn)));
        [HLn, ~, out] = rpca_pcp(Hn, lam_pcp, opts); 
        HL = HLn * scaleH;

        % De-Hankelize -> Δx̂ , integrate -> x̂, robust 2λ alignment
        dy_hat = dehankel_to_vector(HL);
        x_hat  = [y0(1), y0(1) + cumsum(dy_hat)];
        kstar  = median( round( (x_hat - y0)/(2*lambda) ) );
        x_hat  = x_hat - 2*lambda*kstar;

        % Quality
        snr_list(ii) = 20*log10( max(norm(x0),eps) / max(norm(x0 - x_hat),eps) );
        fprintf('  L=%3d | %dx%d | relres=%.2e | SNR=%.2f dB\n', ...
                L, size(H,1), size(H,2), out.relres, snr_list(ii));

        if snr_list(ii) > best.snr
            best = struct('L',L,'x_hat',x_hat,'dy_hat',dy_hat,'HL',HL,'snr',snr_list(ii));
        end
    end

    % Plot best recovery
    figure('Name','Recovery at fs0 (best L)','Color','w');
    plot(t0, x0, 'LineWidth',1.25); hold on;
    plot(t0, best.x_hat, '--', 'LineWidth',1.25); grid on;
    xlabel('t [s]'); ylabel('amplitude');
    title(sprintf('Best recovery at fs=%.0f Hz | L=%d | SNR=%.2f dB', fs0, best.L, best.snr));
    legend('x','\hat{x}','Location','best');

    % Plot effect of L (if multiple)
    if numel(L_list0)>1
        figure('Name','Effect of Hankel height (fs0)','Color','w');
        plot(L_list0, snr_list, 'o-','LineWidth',1.25); grid on;
        xlabel('Hankel height L'); ylabel('SNR (dB)');
        title('Recovery SNR vs L (fixed T, baseline fs)');
    end
end

%% -------- 3) Sweep fs with fixed T (N varies, even if N<40) --------
fs_min = 240; fs_max = 2e4; npts = 200;
fs_vec = round(logspace(log10(fs_min), log10(fs_max), npts));

err_rel = nan(size(fs_vec));
snr_db  = nan(size(fs_vec));
wrap_pct = nan(size(fs_vec));

fprintf('\n--- Sweep fs with fixed T=%.4fs (tiny-N enabled) ---\n', T);
for kk=1:numel(fs_vec)
    fs_k = fs_vec(kk);
    N_k  = max(8, round(T*fs_k));     % allow very small N
    t_k  = (0:N_k-1)/fs_k;

    % signal
    if USE_TWO_TONE
        x_k = A1*cos(2*pi*f1*t_k + phi1) + A2*cos(2*pi*f2*t_k + phi2);
    else
        x_k = A1*cos(2*pi*f1*t_k + phi1);
    end

    % fold
    y_k = fold_centered(x_k, lambda);

    % wrap density
    ktrue_k = round((x_k - y_k)/(2*lambda));
    wraps_k = find(diff(ktrue_k)~=0);
    wrap_pct(kk) = 100*numel(wraps_k)/N_k;

    % Hankel(Δy) with feasible L
    dy_k = diff(y_k);
    m_k  = numel(dy_k);
    [L_listk, feasible] = pick_L_list(m_k, r_est);
    if ~feasible
        % not enough samples for even minimal Hankel — skip metrics
        continue;
    end
    % pick a balanced feasible L (middle of feasible range)
    L_k = L_listk(round(numel(L_listk)/2));

    H_k = hankel_from_vector(dy_k, L_k);
    scaleH = median(abs(H_k(:)) + eps);
    Hn_k = H_k / scaleH;

    % PCP
    lam_pcp = 1/sqrt(max(size(Hn_k)));
    [HLn_k, ~, outk] = rpca_pcp(Hn_k, lam_pcp, opts); 
    HL_k = HLn_k * scaleH;

    % Back to signal
    dy_hat_k = dehankel_to_vector(HL_k);
    x_hat_k  = [y_k(1), y_k(1) + cumsum(dy_hat_k)];
    kstar_k  = median( round( (x_hat_k - y_k)/(2*lambda) ) );
    x_hat_k  = x_hat_k - 2*lambda*kstar_k;

    % metrics
    err_rel(kk) = norm(x_k - x_hat_k) / max(norm(x_k), eps);
    snr_db(kk)  = 20*log10( max(norm(x_k),eps) / max(norm(x_k - x_hat_k),eps) );
end

% Plots: error & wrap density vs fs
figure('Name','Error vs Sampling (fixed T, tiny-N OK)','Color','w');

subplot(1,2,1);
semilogx(fs_vec, err_rel, 'o-','LineWidth',1.25); grid on; hold on;
yyaxis right; semilogx(fs_vec, wrap_pct, 's--','LineWidth',1.1);
ylabel('Wrap density (%)');
yyaxis left; xlabel('Sampling frequency f_s (Hz)');
ylabel('Relative error  ||x-\hat{x}||/||x||');
title(sprintf('Fixed T=%.4fs: error & wrap density vs f_s', T));
legend('Rel. error','Wrap density','Location','best');

subplot(1,2,2);
semilogx(fs_vec, snr_db, 'o-','LineWidth',1.25); grid on;
xlabel('Sampling frequency f_s (Hz)');
ylabel('SNR (dB)');
title('Fixed T: recovery SNR vs f_s');

end

%% ================= helpers =================
function y = fold_centered(x, lambda)
    y = mod(x + lambda, 2*lambda) - lambda;
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

% ---- choose feasible Hankel heights for tiny m = length(Δy) ----
function [L_list, feasible] = pick_L_list(m, r_est)
% Need L>=Lmin and M=m-L+1 >= Lmin, where Lmin = max(2, r_est+1).
    Lmin = max(2, r_est+1);
    Lmax = m - Lmin + 1;
    feasible = (Lmax >= Lmin);
    if ~feasible
        L_list = [];
        return;
    end
    cand = unique(round([0.3 0.4 0.5 0.6 0.7]*m));
    cand = cand(cand >= Lmin & cand <= Lmax);
    if isempty(cand)
        cand = floor((Lmin + Lmax)/2);
    end
    % include endpoints for robustness on tiny sizes
    L_list = unique([Lmin, cand, Lmax]);
end

%% -------- Find fs where error < 0.05% and report wrap density ----------
target_err = 5e-4;   % 0.05%
valid = ~isnan(err_rel);
candidates = find(valid & err_rel <= target_err);

if isempty(candidates)
    fprintf('[TARGET] No fs in the current sweep met rel. error < %.4g.\n', target_err);
    fprintf('         Try increasing fs_max or lambda, or refine with a denser sweep.\n');
else
    % pick the smallest fs that meets the target
    [~, idx_in_candidates_sorted] = min(fs_vec(candidates));
    idx_star = candidates(idx_in_candidates_sorted);

    fs_star    = fs_vec(idx_star);
    err_star   = err_rel(idx_star);
    snr_star   = snr_db(idx_star);
    wrap_star  = wrap_pct(idx_star);   % in percent
    N_star     = round(T * fs_star);

    fprintf('[TARGET] Achieved rel. error %.4g at fs = %.0f Hz (N = %d samples)\n', ...
            err_star, fs_star, N_star);
    fprintf('         Wrap density at that fs: %.2f%%   |   SNR: %.2f dB\n', ...
            wrap_star, snr_star);

    % annotate on the figures if they exist
    try
        figure(findobj('Name','Error vs Sampling (fixed T)')); % your earlier name
    catch
    end
    if exist('fs_vec','var')
        % Left subplot: error + wrap density vs fs
        subplot(1,2,1); hold on;
        xline(fs_star,'--');
        text(fs_star, max(err_rel,[],'omitnan'), sprintf('  f_s^*=%dHz', round(fs_star)), ...
             'Rotation',90,'VerticalAlignment','bottom');

        % Right subplot: SNR vs fs
        subplot(1,2,2); hold on;
        xline(fs_star,'--');
        text(fs_star, max(snr_db,[],'omitnan'), sprintf('  f_s^*=%dHz', round(fs_star)), ...
             'Rotation',90,'VerticalAlignment','bottom');
    end
end
