function demo_relaxed_integer_rpca()
% DEMO_RELAXED_INTEGER_RPCA  Demonstration of RPCA with relaxed integer constraint
% for recovering bandlimited signals from modulo-folded observations.
%
% This script:
% 1) Generates a multitone signal x(t)
% 2) Folds it into [-lambda, lambda) to get y[n]
% 3) Applies RPCA on Hankel(Δy) with the new relaxed integer constraint
% 4) Compares against standard RPCA (no integer awareness)
% 5) Shows recovery quality metrics
%
% WHY THE ORIGINAL APPROACH FAILED:
% ===================================
% 
% The original `Integer_round_PCP.m` used HARD ROUNDING inside the IALM loop:
%   Sbar = round(Sbar_cont);
%
% This failed because:
%
% 1. DISCONTINUITY BREAKS CONVERGENCE
%    - IALM/ADMM requires smooth proximal operators
%    - round() is a step function (discontinuous)
%    - The algorithm can oscillate: S jumps between integers, L adjusts,
%      then S jumps again, never converging
%
% 2. NO GRADIENT INFORMATION
%    - Hard rounding has zero derivative almost everywhere
%    - The optimizer can't "feel" how close it is to the correct integer
%    - It's like navigating blindfolded
%
% 3. HANKEL STRUCTURE INCONSISTENCY
%    - After rounding S, anti-diagonal averaging in de-Hankelization produces
%      NON-INTEGER values for Δs
%    - Example: If Hankel(s) has entries [1, 1, 2] on an anti-diagonal,
%      the averaged value is 4/3 ≈ 1.33, not an integer!
%
% 4. POST-ITERATION ROUNDING DESTROYS LOW-RANK STRUCTURE
%    - In `Last_rounding.m`: L := M - round(S)
%    - The SVT-recovered L had specific low-rank structure
%    - Recomputing L = M - round(S) gives an arbitrary matrix that is
%      generally NOT low-rank anymore
%
% WHY OPTION A (RELAXED INTEGER) WORKS:
% =====================================
%
% 1. SMOOTH PENALTY PRESERVES CONVERGENCE
%    - Phi_int(s) = 1 - exp(-(s - round(s))^2 / (2*sigma^2))
%    - This is smooth and differentiable everywhere
%    - Gradient: grad = (s - round(s))/sigma^2 * exp(...)
%    - IALM can use this gradient to smoothly guide S toward integers
%
% 2. GRADUAL APPROACH TO INTEGERS
%    - The algorithm doesn't jump to integers; it slides toward them
%    - If the data doesn't support a particular integer, S can stay
%      fractional (graceful degradation)
%
% 3. LOW-RANK STRUCTURE PRESERVED
%    - We NEVER recompute L from a modified S
%    - L is always the output of proper SVT
%    - The integer encouragement happens independently in the S update
%
% 4. HANDLES NOISE GRACEFULLY
%    - If observations are noisy, forcing exact integers would be wrong
%    - The soft penalty allows S ≈ 2.1 when the true value is 2 but
%      noise pushes it slightly away
%
% Author: Generated for BTP-2 research
% Date: 2026-01-30

clear; close all; clc;
rng(42);  % For reproducibility

%% ==================== USER PARAMETERS ====================
% Signal parameters
component_freqs = [120, 280, 450];   % Hz (multitone frequencies)
component_amps  = [2.0, 1.5, 1.0];   % Amplitudes
component_phs   = [0, 0.3*pi, -0.2*pi];  % Phases

% Sampling and timing
FS_CONT  = 200e3;   % High-rate "continuous" proxy (Hz)
FS_SAMPLE = 8000;   % Actual sampling rate for RPCA (Hz)
DURATION = 0.03;    % Signal duration (seconds)

% Folding parameter
lambda = 1.8;       % Folding range: [-lambda, lambda)

% Zoom window for detailed plots
zoom_t0 = 0.005;
zoom_t1 = 0.015;

% RPCA options
opts_standard = struct('tol', 1e-7, 'max_iter', 500, 'verbose', 0);
opts_relaxed  = struct('tol', 1e-7, 'max_iter', 500, 'verbose', 1, ...
                       'beta', 0.15, 'sigma', 0.25, 'beta_schedule', 'increasing');

%% ==================== 1) GENERATE CONTINUOUS SIGNAL ====================
fprintf('=============================================================\n');
fprintf('DEMO: Relaxed Integer RPCA for Modulo Folding Recovery\n');
fprintf('=============================================================\n\n');

t_cont = 0 : 1/FS_CONT : DURATION - 1/FS_CONT;
N_cont = numel(t_cont);

% Build multitone signal
x_cont = zeros(1, N_cont);
for k = 1:numel(component_freqs)
    x_cont = x_cont + component_amps(k) * ...
             cos(2*pi*component_freqs(k)*t_cont + component_phs(k));
end

% Fold the continuous signal
x_cont_fold = fold_centered(x_cont, lambda);

fprintf('Signal: %d-tone, freqs = [%s] Hz\n', numel(component_freqs), ...
        num2str(component_freqs));
fprintf('Folding lambda = %.2f, range [%.2f, %.2f)\n', lambda, -lambda, lambda);
fprintf('Duration = %.3fs, FS_CONT = %.0f Hz, FS_SAMPLE = %.0f Hz\n\n', ...
        DURATION, FS_CONT, FS_SAMPLE);

%% ==================== 2) SAMPLE FROM CONTINUOUS ====================
t_samp = 0 : 1/FS_SAMPLE : DURATION - 1/FS_SAMPLE;
N_samp = numel(t_samp);

% Find indices in continuous grid
idx_samp = round(t_samp * FS_CONT) + 1;
idx_samp = min(max(idx_samp, 1), N_cont);

% Sample both signals
x_samp      = x_cont(idx_samp);       % Ground truth (unfolded)
y_samp_fold = x_cont_fold(idx_samp);  % Observed (folded)

% Compute wrap statistics
k_true = round((x_samp - y_samp_fold) / (2*lambda));
wrap_locations = find(diff(k_true) ~= 0);
wrap_density = 100 * numel(wrap_locations) / N_samp;

fprintf('Sampled N = %d points\n', N_samp);
fprintf('Wrap locations: %d (%.2f%% wrap density)\n\n', ...
        numel(wrap_locations), wrap_density);

%% ==================== 3) BUILD HANKEL MATRIX ====================
% Work on first differences
Dy = diff(y_samp_fold);
m = numel(Dy);

% Choose Hankel height (near-balanced)
L = floor(0.5 * m);
H = hankel(Dy(1:L), Dy(L:m));
[nRows, nCols] = size(H);

fprintf('Hankel matrix: %d x %d (from Δy of length %d)\n', nRows, nCols, m);

% Normalize and scale
scaleH = median(abs(H(:))) + eps;
alpha = 2 * lambda;
Hn = H / scaleH;
Mbar = Hn / alpha;

fprintf('Scaling: scaleH = %.4f, alpha = 2*lambda = %.2f\n\n', scaleH, alpha);

%% ==================== 4) RUN BOTH RPCA METHODS ====================
lambda_pcp = 1 / sqrt(max(size(Mbar)));

fprintf('--- Running Standard RPCA (no integer awareness) ---\n');
tic;
[HLbar_std, HSbar_std, out_std] = rpca_pcp_standard(Mbar, lambda_pcp, opts_standard);
time_std = toc;
fprintf('Standard RPCA: %d iters, %.3f sec, rank(L)=%d, nnz(S)=%d\n\n', ...
        out_std.iter, time_std, out_std.rankL, out_std.nnzS);

fprintf('--- Running Relaxed Integer RPCA ---\n');
tic;
[HLbar_rel, HSbar_rel, out_rel] = rpca_relaxed_integer(Mbar, lambda_pcp, opts_relaxed);
time_rel = toc;
fprintf('Relaxed RPCA: %d iters, %.3f sec, rank(L)=%d, nnz(S)=%d\n', ...
        out_rel.iter, time_rel, out_rel.rankL, out_rel.nnzS);
fprintf('Integrality score: %.4f\n\n', out_rel.S_integrality);

%% ==================== 5) RECOVER SIGNALS ====================
% Standard RPCA recovery
HLn_std = alpha * HLbar_std;
HL_std = HLn_std * scaleH;
Dx_hat_std = dehankel_to_vector(HL_std);
x_hat_std = integrate_from_diffs(y_samp_fold, Dx_hat_std);
kstar_std = median(round((x_hat_std - y_samp_fold) / (2*lambda)));
x_hat_std = x_hat_std - 2*lambda*kstar_std;

% Relaxed RPCA recovery
HLn_rel = alpha * HLbar_rel;
HL_rel = HLn_rel * scaleH;
Dx_hat_rel = dehankel_to_vector(HL_rel);
x_hat_rel = integrate_from_diffs(y_samp_fold, Dx_hat_rel);
kstar_rel = median(round((x_hat_rel - y_samp_fold) / (2*lambda)));
x_hat_rel = x_hat_rel - 2*lambda*kstar_rel;

%% ==================== 6) COMPUTE METRICS ====================
% Relative errors
err_std = norm(x_samp - x_hat_std) / norm(x_samp);
err_rel = norm(x_samp - x_hat_rel) / norm(x_samp);

% SNR
snr_std = 20 * log10(norm(x_samp) / max(norm(x_samp - x_hat_std), eps));
snr_rel = 20 * log10(norm(x_samp) / max(norm(x_samp - x_hat_rel), eps));

fprintf('=============================================================\n');
fprintf('RESULTS COMPARISON\n');
fprintf('=============================================================\n');
fprintf('                    Standard RPCA    Relaxed Integer RPCA\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Relative Error:     %.6f          %.6f\n', err_std, err_rel);
fprintf('SNR (dB):           %.2f              %.2f\n', snr_std, snr_rel);
fprintf('Iterations:         %d                %d\n', out_std.iter, out_rel.iter);
fprintf('Time (sec):         %.3f             %.3f\n', time_std, time_rel);
fprintf('=============================================================\n\n');

% Check integrality of sparse components
S_std_scaled = HSbar_std;
S_rel_scaled = HSbar_rel;
int_dev_std = mean((S_std_scaled(:) - round(S_std_scaled(:))).^2);
int_dev_rel = mean((S_rel_scaled(:) - round(S_rel_scaled(:))).^2);

fprintf('Integrality Analysis (in scaled domain):\n');
fprintf('  Standard: mean squared deviation from integers = %.6f\n', int_dev_std);
fprintf('  Relaxed:  mean squared deviation from integers = %.6f\n', int_dev_rel);
fprintf('  Improvement: %.1fx closer to integers\n\n', int_dev_std / max(int_dev_rel, eps));

%% ==================== 7) PLOTS ====================
% Figure 1: Original vs Folded (continuous)
figure('Name', 'Continuous-Time Signal', 'Color', 'w', 'Position', [50 500 1200 300]);
plot(t_cont, x_cont, 'LineWidth', 1.0); hold on;
plot(t_cont, x_cont_fold, '--', 'LineWidth', 1.0);
yline(lambda, 'k--', 'LineWidth', 0.5);
yline(-lambda, 'k--', 'LineWidth', 0.5);
grid on; xlim([0 DURATION]);
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Continuous-Time Signal | \\lambda = %.2f | %d tones', lambda, numel(component_freqs)));
legend('x(t) original', 'y(t) folded', 'Location', 'best');

% Figure 2: Sampled signals (zoom)
figure('Name', 'Sampled Signals (Zoom)', 'Color', 'w', 'Position', [50 150 1200 300]);
zi = (t_samp >= zoom_t0) & (t_samp <= zoom_t1);
stem(t_samp(zi), x_samp(zi), 'filled', 'MarkerSize', 4); hold on;
stem(t_samp(zi), y_samp_fold(zi), 'filled', 'MarkerSize', 4);
grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Sampled Signals (%.3f-%.3f s) | fs = %.0f Hz', zoom_t0, zoom_t1, FS_SAMPLE));
legend('x[n] original', 'y[n] folded');

% Figure 3: Recovery comparison
figure('Name', 'Recovery Comparison', 'Color', 'w', 'Position', [100 50 1400 400]);

subplot(1,2,1);
plot(t_samp, x_samp, 'LineWidth', 1.5); hold on;
plot(t_samp, x_hat_std, '--', 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Standard RPCA | SNR = %.2f dB | RelErr = %.2e', snr_std, err_std));
legend('x (ground truth)', '\hat{x} (recovered)', 'Location', 'best');

subplot(1,2,2);
plot(t_samp, x_samp, 'LineWidth', 1.5); hold on;
plot(t_samp, x_hat_rel, '--', 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Relaxed Integer RPCA | SNR = %.2f dB | RelErr = %.2e', snr_rel, err_rel));
legend('x (ground truth)', '\hat{x} (recovered)', 'Location', 'best');

% Figure 4: Sparse component integrality
figure('Name', 'Sparse Component Analysis', 'Color', 'w', 'Position', [150 100 1000 400]);

subplot(1,2,1);
histogram(S_std_scaled(:), 50, 'Normalization', 'probability');
hold on; xline(round(min(S_std_scaled(:))):round(max(S_std_scaled(:))), 'r--', 'LineWidth', 0.5);
xlabel('S_{bar} value'); ylabel('Probability');
title('Standard RPCA: S distribution');
grid on;

subplot(1,2,2);
histogram(S_rel_scaled(:), 50, 'Normalization', 'probability');
hold on; xline(round(min(S_rel_scaled(:))):round(max(S_rel_scaled(:))), 'r--', 'LineWidth', 0.5);
xlabel('S_{bar} value'); ylabel('Probability');
title('Relaxed Integer RPCA: S distribution');
grid on;

% Figure 5: Error over samples
figure('Name', 'Sample-wise Error', 'Color', 'w', 'Position', [200 150 1200 300]);
plot(t_samp, abs(x_samp - x_hat_std), 'LineWidth', 1.0); hold on;
plot(t_samp, abs(x_samp - x_hat_rel), 'LineWidth', 1.0);
grid on; xlabel('Time (s)'); ylabel('|Error|');
title('Sample-wise Absolute Error');
legend(sprintf('Standard (mean=%.4f)', mean(abs(x_samp - x_hat_std))), ...
       sprintf('Relaxed (mean=%.4f)', mean(abs(x_samp - x_hat_rel))));

fprintf('Done. Figures generated.\n');

end

%% ==================== HELPER FUNCTIONS ====================

function y = fold_centered(x, lambda)
% Centered modulo folding into [-lambda, lambda)
    y = mod(x + lambda, 2*lambda) - lambda;
end

function v = dehankel_to_vector(H)
% Anti-diagonal averaging to convert Hankel matrix back to vector
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

function x_hat = integrate_from_diffs(y, D_hat)
% Integrate first differences to recover signal
% Uses first sample of y as initial condition
    x_hat = [y(1), y(1) + cumsum(D_hat)];
end

function [L, S, out] = rpca_pcp_standard(M, lambda, opts)
% Standard RPCA via PCP (IALM) - no integer awareness
    if ~isfloat(M), M = double(M); end
    [m, n] = size(M);
    
    if ~exist('lambda','var') || isempty(lambda)
        lambda = 1/sqrt(max(m,n));
    end
    if ~exist('opts','var') || isempty(opts), opts = struct(); end
    
    tol = get_opt(opts, 'tol', 1e-7);
    max_iter = get_opt(opts, 'max_iter', 1000);
    verbose = get_opt(opts, 'verbose', 1);
    mu = get_opt(opts, 'mu', []);
    mu_factor = get_opt(opts, 'mu_factor', 1.5);
    mu_max = get_opt(opts, 'mu_max', 1e7);
    
    normM = norm(M, 'fro');
    if normM == 0
        L = zeros(m,n); S = zeros(m,n);
        out = struct('iter',0,'relres',0,'obj',0,'rankL',0,'nnzS',0);
        return;
    end
    
    if isempty(mu)
        try smax = svds(M,1); catch, smax = norm(M,2); end
        mu = 1.25 / max(smax, eps);
    end
    
    L = zeros(m,n);
    S = zeros(m,n);
    Y = M / max(norm(M,2), eps);
    
    relres = inf; k = 0; r = 0; obj = NaN;
    
    while (relres > tol) && (k < max_iter)
        k = k + 1;
        
        % L update (SVT)
        W = M - S + (1/mu)*Y;
        [U, Sig, V] = svd(W, 'econ');
        s = diag(Sig);
        s_shr = max(s - 1/mu, 0);
        r = nnz(s_shr > 0);
        if r > 0
            L = U(:,1:r) * diag(s_shr(1:r)) * V(:,1:r)';
        else
            L = zeros(m,n);
        end
        
        % S update (soft threshold)
        T = M - L + (1/mu)*Y;
        S = sign(T) .* max(abs(T) - lambda/mu, 0);
        
        % Dual update
        Z = M - L - S;
        Y = Y + mu*Z;
        
        relres = norm(Z, 'fro') / normM;
        obj = sum(s_shr) + lambda * sum(abs(S(:)));
        
        mu = min(mu * mu_factor, mu_max);
    end
    
    out.iter = k;
    out.relres = relres;
    out.obj = obj;
    out.rankL = r;
    out.nnzS = nnz(abs(S) > 1e-6);
end

function v = get_opt(opts, field, defaultv)
    if isfield(opts, field) && ~isempty(opts.(field))
        v = opts.(field);
    else
        v = defaultv;
    end
end
