function demo_unlimited_rpca_hankel()
rng(0);
% Sampling
fs = 3000;
Ts = 1/fs;       % sampling time (seconds)
N  = 256;         % number of samples
t  = (0:N-1)*Ts;
A1 = 2.0;  f1 = 120;   phi1 = 0.30*pi;
A2 = 1.2;  f2 = 280;   phi2 = -0.55*pi;

%x = A1*cos(2*pi*f1*t + phi1) + A2*cos(2*pi*f2*t + phi2); %Dual Tone
x = A1 *cos(2*pi*f1*t); % Mono Tone
%m=0.7; fm=30; fc=150; x = (1+m*cos(2*pi*fm*t)).*cos(2*pi*fc*t); % AM Modulated Wave
%beta=0.5; fm=30; fc=200; x = 1.5*cos(2*pi*fc*t + beta*sin(2*pi*fm*t)); 
% FM Modulated Wave

% Folding (self-reset ADC) threshold 
lambda = 1;        

% Robust PCA parameters
opts = struct('tol',1e-7,'max_iter',1000,'verbose',1);

%Folding
y = fold_centered(x, lambda);    % y in [-lambda, lambda]

% Original Vs Fold
figure('Name','Original vs Folded','Color','w'); 
subplot(2,1,1); plot(t,x,'LineWidth',1); grid on;
title('Original signal x(t)'); xlabel('t [s]'); ylabel('amplitude');

subplot(2,1,2); plot(t,y,'LineWidth',1); grid on;
title(sprintf('Folded signal y(t) in [-\\lambda,\\lambda], \\lambda=%.3g',lambda));
xlabel('t [s]'); ylabel('amplitude');

%% -------  Hankel lifting of first-difference --------

dx = diff(x);                 % ground-truth difference (for evaluation)
dy = diff(y);                 % observed (folded) difference

% Candidate Hankel heights to show effect of matrix size
L_list = unique(round([0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.80, 0.9] * numel(dy)));
L_list(L_list < 20) = [];                 % keep sensible sizes
L_list(L_list > numel(dy)-20) = [];

snr_list = zeros(size(L_list));
best = struct();

fprintf('\n--- RPCA on Hankel(dy): sweeping Hankel height L ---\n');
for ii = 1:numel(L_list)
    L = L_list(ii);
    H = hankel_from_vector(dy, L);       % L x M Hankel matrix

    % RPCA (PCP)
    lam_rpca = 1/sqrt(max(size(H)));     % standard default
    [H_L, H_S, out] = rpca_pcp(H, lam_rpca, opts);

    % De-Hankelization (diagonal averaging) to return to a 1D sequence
    dy_hat = dehankel_to_vector(H_L);

    % Integrate once to get signal (up to constant): set initial value via y(1)
    x_hat = [y(1), y(1) + cumsum(dy_hat)];  % length N

    % Quality (vs ground truth)
    snr_list(ii) = snr(x, x - x_hat);

    fprintf('  L=%4d | size(H)=%dx%d | rank(L)=? | relres=%.2e | SNR=%.2f dB\n',...
            L, size(H,1), size(H,2), out.relres, snr_list(ii));

    % keep best
    if ii==1 || snr_list(ii) > best.snr
        best = struct('L',L,'x_hat',x_hat,'dy_hat',dy_hat,'H_L',H_L,'H_S',H_S,...
                      'relres',out.relres,'snr',snr_list(ii),'H',H);
    end
end

%% ----------- Show best recovery ---------
figure('Name','Recovery (best Hankel size)','Color','w');
plot(t, x, 'LineWidth',1.25); hold on;
plot(t, best.x_hat, '--', 'LineWidth',1.25);
grid on; xlabel('t [s]'); ylabel('amplitude');
title(sprintf('Recovery using RPCA on Hankel(dy) | best L=%d | SNR=%.2f dB', ...
      best.L, best.snr));
legend('Original x','Recovered \hat{x}','Location','best');

% Also show sparse part as an event detector (folding spikes)
%figure('Name','Sparse part (event structure)','Color','w');
%imagesc(abs(best.H_S)>1e-6); axis image; colormap(gray);
%title('Support of sparse component |H\_S|>0 (fold event structure)');
%xlabel('columns'); ylabel('rows');

%% ---------- Show Hankel size effect ----------
figure('Name','Effect of Hankel height','Color','w');
plot(L_list, snr_list, 'o-','LineWidth',1.25); grid on;
xlabel('Hankel height L'); ylabel('SNR of recovered x (dB)');
title('Recovery quality vs Hankel size');
fprintf('\nBest L = %d, Recovery SNR = %.2f dB\n', best.L, best.snr);

end

% ====== Functions====

function y = fold_centered(x, lambda)
    y = mod(x + lambda, 2*lambda) - lambda;
end

function H = hankel_from_vector(v, L)
% Build L x M Hankel matrix from vector v of length N: anti-diagonals share v.
% H(i,j) = v(i+j-1), i=1..L, j=1..M, where M = N-L+1
    N = numel(v);
    M = N - L + 1;
    if M <= 0
        error('Hankel height L too large for vector length %d.', N);
    end
    H = hankel(v(1:L), v(L:N));
end

function v = dehankel_to_vector(H)
% Diagonal averaging (a.k.a. Hankelization) to map LxM Hankel back to vector
% Output length is L+M-1, matching original length used to build H.
    [L,M] = size(H);
    Lm = L + M - 1;
    v = zeros(1, Lm);
    counts = zeros(1, Lm);
    for i = 1:L
        for j = 1:M
            k = i + j - 1;
            v(k) = v(k) + H(i,j);
            counts(k) = counts(k) + 1;
        end
    end
    v = v ./ counts;
end

% ==================== RPCA (PCP via IALM) ====================
function [L,S,out] = rpca_pcp(M, lambda, opts)
% Inexact Augmented Lagrange Multiplier (IALM) for:
%   min ||L||_* + lambda||S||_1  s.t. L+S = M
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

    normM = norm(M,'fro'); 
    if normM==0
        L=zeros(m,n); S=zeros(m,n);
        out=struct('iter',0,'relres',0,'obj',0,'rankL',0,'nnzS',0); 
        return; 
    end
    if isempty(mu)
        try smax = svds(M,1); catch, smax = norm(M,2); end
        mu = 1.25 / max(smax, eps);
    end

    L = zeros(m,n); S = zeros(m,n);
    Y = M / max(norm(M,2), eps);
    relres = inf; k = 0; r = 0; obj = NaN;

    if verbose
        fprintf('RPCA-PCP: lambda=%.3g, tol=%.1e\n', lambda, tol);
        fprintf('%5s  %10s  %10s  %8s  %8s  %8s\n', ...
                'iter','relres','obj','rank(L)','nnz(S)','mu');
    end

    while (relres > tol) && (k < max_iter)
        k = k + 1;

        % --- SVT update for L ---
        W = M - S + (1/mu)*Y;
        [U,Sig,V] = svd(W,'econ');
        s = diag(Sig);
        s_shr = max(s - 1/mu, 0);
        r = nnz(s_shr>0);
        if r>0
            L = U(:,1:r) * diag(s_shr(1:r)) * V(:,1:r)';
        else
            L = zeros(m,n);
        end

        % --- Soft-thresh for S ---
        T = M - L + (1/mu)*Y;
        S = sign(T).*max(abs(T) - lambda/mu, 0);

        % --- Dual & diagnostics ---
        Z = M - L - S;
        Y = Y + mu * Z;
        relres = norm(Z,'fro')/normM;
        obj = sum(s_shr) + lambda*sum(abs(S(:)));
        if verbose && (mod(k,10)==0 || k==1)
            fprintf('%5d  %10.3e  %10.4e  %8d  %8d  %8.2e\n', ...
                    k, relres, obj, r, nnz(S), mu);
        end
        mu = min(mu * mu_factor, mu_max);
    end
    if verbose
        fprintf('Done in %d iters, relres=%.3e, rank(L)=%d, nnz(S)=%d\n', ...
                 k, relres, r, nnz(S));
    end
    out = struct('iter',k,'relres',relres,'obj',obj,'rankL',r,'nnzS',nnz(S));
end

function v = get_opt(opts, field, defaultv)
    if isfield(opts,field) && ~isempty(opts.(field)), v = opts.(field); else, v = defaultv; end
end

%% ------Error vs sampling time sweep (240 Hz - 10 kHz) ---
fs_min = 240;                 % Hz
fs_max = 1e4;                 % Hz
npts   = 100;                  % number of sweep points (log-spaced)
fs_vec = round(logspace(log10(fs_min), log10(fs_max), npts));
err_rel = zeros(size(fs_vec));
snr_db  = zeros(size(fs_vec));
wrap_pct = zeros(size(fs_vec));

for kk = 1:numel(fs_vec)
    fs_k = fs_vec(kk);
    Ts_k = 1/fs_k;

    % --- build signal for this Ts_k ---
    t_k = (0:N-1) * Ts_k;

    % Use the SAME model you used above:
    x_k = A1 * cos(2*pi*f1*t_k + phi1);
    % If you want two-tone instead, comment the above line and uncomment:
    % x_k = A1*cos(2*pi*f1*t_k + phi1) + A2*cos(2*pi*f2*t_k + phi2);

    % --- fold, difference, Hankel, RPCA ---
    y_k  = fold_centered(x_k, lambda);
    dy_k = diff(y_k);

    L_k  = floor(0.5 * numel(dy_k));              % balanced Hankel
    H_k  = hankel_from_vector(dy_k, L_k);

    [HL_k, HS_k] = rpca_pcp(H_k, 1/sqrt(max(size(H_k))), opts);

    dy_hat_k = dehankel_to_vector(HL_k);

    % --- integrate and align global 2λ offset (robust median rounding) ---
    x_hat_k = [y_k(1), y_k(1) + cumsum(dy_hat_k)];
    kstar   = median( round( (x_hat_k - y_k) / (2*lambda) ) );
    x_hat_k = x_hat_k - 2*lambda * kstar;

    % --- metrics ---
    err_rel(kk) = norm(x_k - x_hat_k) / max(norm(x_k), eps);
    snr_db(kk)  = 20*log10( max(norm(x_k),eps) / max(norm(x_k - x_hat_k),eps) );

    % wrap density (for intuition)
    k_true_k = round((x_k - y_k) / (2*lambda));
    wraps_k  = find(diff(k_true_k) ~= 0);
    wrap_pct(kk) = numel(wraps_k) / N * 100;
end

% ---- plots: error vs fs and error vs Ts (both log scale) ----
figure('Name','Error vs Sampling','Color','w');

subplot(1,2,1);
semilogx(fs_vec, err_rel, 'o-','LineWidth',1.25); grid on; hold on;
yyaxis right;
semilogx(fs_vec, wrap_pct, 's--','LineWidth',1.0);
ylabel('Wrap density (%)');
yyaxis left;
xlabel('Sampling frequency f_s (Hz)');
ylabel('Relative error  ||x-\hat{x}|| / ||x||');
title('Recovery error vs sampling frequency');
legend('Relative error','Wrap density','Location','best');

subplot(1,2,2);
semilogx(1./fs_vec, err_rel, 'o-','LineWidth',1.25); grid on;
set(gca,'XDir','reverse');  % small Ts (high fs) on the left
xlabel('Sampling time T_s (s)');
ylabel('Relative error  ||x-\hat{x}|| / ||x||');
title('Recovery error vs sampling time');

