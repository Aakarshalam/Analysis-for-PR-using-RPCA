function sweep_relaxed_integer_rpca()
% SWEEP_RELAXED_INTEGER_RPCA  Comprehensive parameter sweep comparing
% standard RPCA vs relaxed integer RPCA across different conditions.
%
% This script performs systematic experiments varying:
% 1. Wrap density (controlled by lambda)
% 2. Sampling rate
% 3. Signal complexity (number of tones)
% 4. Integer penalty parameters (beta, sigma)
%
% Author: Generated for BTP-2 research
% Date: 2026-01-30

clear; close all; clc;
rng(0);

%% ==================== EXPERIMENT SETTINGS ====================
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║   COMPREHENSIVE SWEEP: Standard vs Relaxed Integer RPCA       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Fixed parameters
FS_CONT  = 200e3;   % Continuous proxy rate
DURATION = 0.04;    % Signal duration

% Base signal
base_freqs = [120, 280, 450];
base_amps  = [2.0, 1.5, 1.0];
base_phs   = [0, 0.3*pi, -0.2*pi];

% RPCA options
opts_std = struct('tol', 1e-7, 'max_iter', 500, 'verbose', 0);
opts_rel = struct('tol', 1e-7, 'max_iter', 500, 'verbose', 0, ...
                  'beta', 0.15, 'sigma', 0.25, 'beta_schedule', 'increasing');

%% ==================== EXPERIMENT 1: WRAP DENSITY vs SNR ====================
fprintf('─────────────────────────────────────────────────────────────────\n');
fprintf('EXPERIMENT 1: Effect of Wrap Density (varying lambda)\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

lambda_values = linspace(0.8, 4.0, 15);  % Lower lambda = more wrapping
fs_fixed = 8000;

snr_std_exp1 = zeros(size(lambda_values));
snr_rel_exp1 = zeros(size(lambda_values));
wrap_density_exp1 = zeros(size(lambda_values));

for i = 1:numel(lambda_values)
    lambda = lambda_values(i);
    
    [snr_std, snr_rel, wrap_dens, ~, ~] = run_single_experiment(...
        base_freqs, base_amps, base_phs, lambda, fs_fixed, FS_CONT, DURATION, ...
        opts_std, opts_rel);
    
    snr_std_exp1(i) = snr_std;
    snr_rel_exp1(i) = snr_rel;
    wrap_density_exp1(i) = wrap_dens;
    
    fprintf('  lambda=%.2f: wrap=%.1f%%, SNR_std=%.1f dB, SNR_rel=%.1f dB, Δ=%.1f dB\n', ...
            lambda, wrap_dens, snr_std, snr_rel, snr_rel - snr_std);
end

%% ==================== EXPERIMENT 2: SAMPLING RATE vs SNR ====================
fprintf('\n─────────────────────────────────────────────────────────────────\n');
fprintf('EXPERIMENT 2: Effect of Sampling Rate\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

fs_values = round(logspace(log10(2000), log10(20000), 15));
lambda_fixed = 1.8;

snr_std_exp2 = zeros(size(fs_values));
snr_rel_exp2 = zeros(size(fs_values));
wrap_density_exp2 = zeros(size(fs_values));

for i = 1:numel(fs_values)
    fs = fs_values(i);
    
    [snr_std, snr_rel, wrap_dens, ~, ~] = run_single_experiment(...
        base_freqs, base_amps, base_phs, lambda_fixed, fs, FS_CONT, DURATION, ...
        opts_std, opts_rel);
    
    snr_std_exp2(i) = snr_std;
    snr_rel_exp2(i) = snr_rel;
    wrap_density_exp2(i) = wrap_dens;
    
    fprintf('  fs=%5d Hz: wrap=%.1f%%, SNR_std=%.1f dB, SNR_rel=%.1f dB\n', ...
            fs, wrap_dens, snr_std, snr_rel);
end

%% ==================== EXPERIMENT 3: SIGNAL COMPLEXITY ====================
fprintf('\n─────────────────────────────────────────────────────────────────\n');
fprintf('EXPERIMENT 3: Effect of Signal Complexity (number of tones)\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

n_tones_list = [1, 2, 3, 4, 5, 6, 7, 8];
lambda_fixed = 1.8;
fs_fixed = 10000;

snr_std_exp3 = zeros(size(n_tones_list));
snr_rel_exp3 = zeros(size(n_tones_list));
rank_exp3 = zeros(size(n_tones_list));

for i = 1:numel(n_tones_list)
    K = n_tones_list(i);
    
    % Generate K tones with random frequencies
    freqs = sort(50 + (400-50)*rand(1, K));
    amps = 1 + rand(1, K);
    phs = 2*pi*rand(1, K);
    
    [snr_std, snr_rel, ~, out_std, out_rel] = run_single_experiment(...
        freqs, amps, phs, lambda_fixed, fs_fixed, FS_CONT, DURATION, ...
        opts_std, opts_rel);
    
    snr_std_exp3(i) = snr_std;
    snr_rel_exp3(i) = snr_rel;
    rank_exp3(i) = out_rel.rankL;
    
    fprintf('  K=%d tones: rank(L)=%d, SNR_std=%.1f dB, SNR_rel=%.1f dB\n', ...
            K, out_rel.rankL, snr_std, snr_rel);
end

%% ==================== EXPERIMENT 4: BETA-SIGMA PARAMETER GRID ====================
fprintf('\n─────────────────────────────────────────────────────────────────\n');
fprintf('EXPERIMENT 4: Optimal (beta, sigma) for relaxed integer penalty\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

beta_range = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50];
sigma_range = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50];
lambda_fixed = 1.8;
fs_fixed = 8000;

snr_grid = zeros(numel(beta_range), numel(sigma_range));
int_grid = zeros(numel(beta_range), numel(sigma_range));  % Integrality score

for ib = 1:numel(beta_range)
    for is = 1:numel(sigma_range)
        opts_test = struct('tol', 1e-7, 'max_iter', 500, 'verbose', 0, ...
                           'beta', beta_range(ib), 'sigma', sigma_range(is), ...
                           'beta_schedule', 'increasing');
        
        [~, snr_rel, ~, ~, out_rel] = run_single_experiment(...
            base_freqs, base_amps, base_phs, lambda_fixed, fs_fixed, FS_CONT, DURATION, ...
            opts_std, opts_test);
        
        snr_grid(ib, is) = snr_rel;
        int_grid(ib, is) = out_rel.S_integrality;
    end
end

% Find optimal
[max_snr, max_idx] = max(snr_grid(:));
[opt_ib, opt_is] = ind2sub(size(snr_grid), max_idx);
fprintf('  Optimal: beta=%.2f, sigma=%.2f → SNR=%.1f dB\n', ...
        beta_range(opt_ib), sigma_range(opt_is), max_snr);

%% ==================== GENERATE PLOTS ====================
fprintf('\n─────────────────────────────────────────────────────────────────\n');
fprintf('Generating summary plots...\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

% Figure 1: Wrap density effect
figure('Name', 'Exp 1: Wrap Density Effect', 'Color', 'w', 'Position', [50 400 600 400]);
yyaxis left;
plot(lambda_values, snr_std_exp1, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Standard RPCA');
hold on;
plot(lambda_values, snr_rel_exp1, 's-', 'LineWidth', 1.5, 'DisplayName', 'Relaxed Integer RPCA');
ylabel('SNR (dB)');
yyaxis right;
plot(lambda_values, wrap_density_exp1, 'd--', 'LineWidth', 1.0, 'DisplayName', 'Wrap Density');
ylabel('Wrap Density (%)');
xlabel('\lambda (folding threshold)');
title('Experiment 1: Effect of Wrap Density');
legend('Location', 'best');
grid on;

% Figure 2: Sampling rate effect
figure('Name', 'Exp 2: Sampling Rate Effect', 'Color', 'w', 'Position', [100 350 600 400]);
semilogx(fs_values, snr_std_exp2, 'o-', 'LineWidth', 1.5); hold on;
semilogx(fs_values, snr_rel_exp2, 's-', 'LineWidth', 1.5);
xlabel('Sampling Rate f_s (Hz)');
ylabel('SNR (dB)');
title('Experiment 2: Effect of Sampling Rate');
legend('Standard RPCA', 'Relaxed Integer RPCA', 'Location', 'best');
grid on;

% Figure 3: Signal complexity effect
figure('Name', 'Exp 3: Signal Complexity', 'Color', 'w', 'Position', [150 300 600 400]);
yyaxis left;
bar(n_tones_list, [snr_std_exp3', snr_rel_exp3'], 'grouped');
ylabel('SNR (dB)');
yyaxis right;
plot(n_tones_list, 2*n_tones_list, 'k--', 'LineWidth', 1.5);
hold on; plot(n_tones_list, rank_exp3, 'ko-', 'LineWidth', 1.5, 'MarkerFaceColor', 'k');
ylabel('Rank');
xlabel('Number of Tones K');
title('Experiment 3: Signal Complexity');
legend('Standard RPCA', 'Relaxed Integer', 'Expected rank (2K)', 'Recovered rank', 'Location', 'best');
grid on;

% Figure 4: Parameter grid heatmap
figure('Name', 'Exp 4: Parameter Optimization', 'Color', 'w', 'Position', [200 250 700 500]);
subplot(1,2,1);
imagesc(sigma_range, beta_range, snr_grid);
colorbar; colormap('hot');
xlabel('\sigma'); ylabel('\beta');
title('SNR (dB) vs (\beta, \sigma)');
set(gca, 'YDir', 'normal');
hold on;
plot(sigma_range(opt_is), beta_range(opt_ib), 'g*', 'MarkerSize', 15, 'LineWidth', 2);

subplot(1,2,2);
imagesc(sigma_range, beta_range, int_grid);
colorbar; colormap('parula');
xlabel('\sigma'); ylabel('\beta');
title('Integrality Score vs (\beta, \sigma)');
set(gca, 'YDir', 'normal');

% Figure 5: Summary comparison bar chart
figure('Name', 'Summary Comparison', 'Color', 'w', 'Position', [250 200 500 400]);
categories = {'Low Wrap\n(λ=3.5)', 'Med Wrap\n(λ=1.8)', 'High Wrap\n(λ=1.0)'};
% Get representative values
idx_low = find(lambda_values >= 3.4, 1);
idx_med = find(lambda_values >= 1.7, 1);
idx_high = find(lambda_values >= 0.95, 1);
if isempty(idx_low), idx_low = numel(lambda_values); end
if isempty(idx_high), idx_high = 1; end

snr_summary = [snr_std_exp1(idx_low), snr_rel_exp1(idx_low);
               snr_std_exp1(idx_med), snr_rel_exp1(idx_med);
               snr_std_exp1(idx_high), snr_rel_exp1(idx_high)];

bar(snr_summary);
set(gca, 'XTickLabel', {'Low Wrap', 'Med Wrap', 'High Wrap'});
ylabel('SNR (dB)');
legend('Standard RPCA', 'Relaxed Integer RPCA', 'Location', 'best');
title('Summary: Recovery Quality by Wrap Density');
grid on;

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║                    SWEEP COMPLETED                            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n');

end

%% ==================== SINGLE EXPERIMENT RUNNER ====================
function [snr_std, snr_rel, wrap_dens, out_std, out_rel] = run_single_experiment(...
    freqs, amps, phs, lambda, fs, FS_CONT, DURATION, opts_std, opts_rel)
% Run a single RPCA experiment with given parameters

    % Generate continuous signal
    t_cont = 0 : 1/FS_CONT : DURATION - 1/FS_CONT;
    N_cont = numel(t_cont);
    
    x_cont = zeros(1, N_cont);
    for k = 1:numel(freqs)
        x_cont = x_cont + amps(k) * cos(2*pi*freqs(k)*t_cont + phs(k));
    end
    
    % Fold
    x_cont_fold = mod(x_cont + lambda, 2*lambda) - lambda;
    
    % Sample
    t_samp = 0 : 1/fs : DURATION - 1/fs;
    N_samp = numel(t_samp);
    idx_samp = min(max(round(t_samp * FS_CONT) + 1, 1), N_cont);
    
    x_samp = x_cont(idx_samp);
    y_samp = x_cont_fold(idx_samp);
    
    % Wrap density
    k_true = round((x_samp - y_samp) / (2*lambda));
    wrap_locs = find(diff(k_true) ~= 0);
    wrap_dens = 100 * numel(wrap_locs) / N_samp;
    
    % Build Hankel
    Dy = diff(y_samp);
    m = numel(Dy);
    if m < 20
        snr_std = NaN; snr_rel = NaN;
        out_std = struct('rankL', 0); out_rel = struct('rankL', 0, 'S_integrality', 0);
        return;
    end
    
    L = floor(0.5 * m);
    H = hankel(Dy(1:L), Dy(L:m));
    
    scaleH = median(abs(H(:))) + eps;
    alpha = 2 * lambda;
    Mbar = H / scaleH / alpha;
    
    lambda_pcp = 1 / sqrt(max(size(Mbar)));
    
    % Standard RPCA
    [HLbar_std, ~, out_std] = rpca_pcp_local(Mbar, lambda_pcp, opts_std);
    HL_std = alpha * HLbar_std * scaleH;
    Dx_std = dehankel_local(HL_std);
    x_std = [y_samp(1), y_samp(1) + cumsum(Dx_std)];
    kstar = median(round((x_std - y_samp) / (2*lambda)));
    x_std = x_std - 2*lambda*kstar;
    
    snr_std = 20*log10(norm(x_samp) / max(norm(x_samp - x_std), eps));
    
    % Relaxed Integer RPCA
    [HLbar_rel, ~, out_rel] = rpca_relaxed_integer(Mbar, lambda_pcp, opts_rel);
    HL_rel = alpha * HLbar_rel * scaleH;
    Dx_rel = dehankel_local(HL_rel);
    x_rel = [y_samp(1), y_samp(1) + cumsum(Dx_rel)];
    kstar = median(round((x_rel - y_samp) / (2*lambda)));
    x_rel = x_rel - 2*lambda*kstar;
    
    snr_rel = 20*log10(norm(x_samp) / max(norm(x_samp - x_rel), eps));
end

%% ==================== LOCAL HELPER FUNCTIONS ====================
function v = dehankel_local(H)
    [L, M] = size(H);
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

function [L, S, out] = rpca_pcp_local(M, lambda, opts)
% Local copy of standard RPCA
    [m, n] = size(M);
    if ~exist('lambda','var') || isempty(lambda), lambda = 1/sqrt(max(m,n)); end
    if ~exist('opts','var'), opts = struct(); end
    
    tol = getfield_default(opts, 'tol', 1e-7);
    max_iter = getfield_default(opts, 'max_iter', 500);
    mu_factor = getfield_default(opts, 'mu_factor', 1.5);
    mu_max = getfield_default(opts, 'mu_max', 1e7);
    
    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(m,n); S = zeros(m,n);
        out = struct('iter',0,'relres',0,'rankL',0,'nnzS',0);
        return;
    end
    
    try mu = 1.25/svds(M,1); catch, mu = 1.25/norm(M,2); end
    
    L = zeros(m,n); S = zeros(m,n); Y = M/max(norm(M,2),eps);
    relres = inf; k = 0; r = 0;
    
    while (relres > tol) && (k < max_iter)
        k = k + 1;
        W = M - S + Y/mu;
        [U,Sig,V] = svd(W,'econ');
        s = diag(Sig);
        s_shr = max(s - 1/mu, 0);
        r = nnz(s_shr > 0);
        if r > 0, L = U(:,1:r)*diag(s_shr(1:r))*V(:,1:r)'; else, L = zeros(m,n); end
        
        T = M - L + Y/mu;
        S = sign(T) .* max(abs(T) - lambda/mu, 0);
        
        Z = M - L - S;
        Y = Y + mu*Z;
        relres = norm(Z,'fro')/normM;
        mu = min(mu*mu_factor, mu_max);
    end
    
    out.iter = k; out.relres = relres; out.rankL = r; out.nnzS = nnz(abs(S)>1e-6);
end

function v = getfield_default(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
