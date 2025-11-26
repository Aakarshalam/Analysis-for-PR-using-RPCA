

clear; close all; clc;

%% ---------------- PARAMETERS (EDIT) ----------------
% Signal components
component_freqs = [300, 1200, 3000];   % Hz
component_amps  = [1.0, 0.6, 0.4];
component_phs   = [0, 0.3*pi, -0.2*pi];

FS_CONT = 200e3;     % high-rate "continuous" proxy (Hz)
DUR     = 0.02;      % seconds

FS_SAMPLE   = 6e3;   % low sampling rate (Hz) — try 20e3 (good) or 4e3 (aliasing)
use_first_M = 120;   % how many low-rate samples go into Hankel (<= total)

% Hankel size N: require 2N-1 <= M_used
N_user = [];         % [] => use floor((M_used+1)/2); else clamped to valid

% PCA retention (choose ONE mode)
PCA_KEEP_K        = 6;      % keep top-K PCs (set [] to use variance mode)
PCA_KEEP_VAR_FRAC = [];     % e.g., 0.99 for 99% variance (set [] if using K)

% ----- Noise controls -----
ADD_NOISE     = true;      % toggle noise on/off
NOISE_MODE    = "snr";     % "snr" or "sigma"
SNR_dB        = 5;        % used if NOISE_MODE="snr": target SNR of noisy low-rate samples
NOISE_SIGMA   = 0.15;      % used if NOISE_MODE="sigma": additive white Gaussian std
NOISE_SEED    = 123;       % set [] for non-deterministic; otherwise reproducible

% Plot options
zoom_samples = 80;          % number of initial samples for zoom plots

%% ---------------- BUILD HIGH-RATE + SAMPLE ----------------
t_cont = 0:1/FS_CONT:DUR - 1/FS_CONT;
x_cont = zeros(size(t_cont));
for k = 1:numel(component_freqs)
    x_cont = x_cont + component_amps(k)*sin(2*pi*component_freqs(k)*t_cont + component_phs(k));
end

t_low = 0:1/FS_SAMPLE:DUR - 1/FS_SAMPLE;
x_low = zeros(size(t_low));
for k = 1:numel(component_freqs)
    x_low = x_low + component_amps(k)*sin(2*pi*component_freqs(k)*t_low + component_phs(k));
end

M_total = numel(x_low);
M_used  = min(use_first_M, M_total);
x_used_clean = x_low(1:M_used);

%% ---------------- ADD NOISE TO LOW-RATE SAMPLES ----------------
if ~isempty(NOISE_SEED); rng(NOISE_SEED); end

switch NOISE_MODE
    case "snr"
        % Target SNR (dB) relative to the CLEAN segment used in Hankel
        sig_power = var(x_used_clean, 1);                 % population var
        noise_power = sig_power / (10^(SNR_dB/10));
        sigma = sqrt(noise_power);
    case "sigma"
        sigma = NOISE_SIGMA;
    otherwise
        error('NOISE_MODE must be "snr" or "sigma".');
end

if ADD_NOISE
    noise_full = sigma * randn(size(x_low));
    x_low_noisy = x_low + noise_full;
else
    x_low_noisy = x_low;
end

x_used_noisy = x_low_noisy(1:M_used);

% Report input SNR on the used segment
if ADD_NOISE
    in_noise_seg = x_used_noisy - x_used_clean;
    SNR_in_dB = 10*log10( var(x_used_clean,1) / var(in_noise_seg,1) );
else
    SNR_in_dB = Inf;
end

%% ---------------- CHOOSE HANKEL SIZE ----------------
if isempty(N_user)
    N = floor((M_used+1)/2);
else
    N = min(N_user, floor((M_used+1)/2));
end
assert(2*N-1 <= M_used, 'Need 2N-1 <= M_used.');

fprintf('M=%d low-rate samples used, Hankel size N=%d (2N-1=%d)\n', M_used, N, 2*N-1);
if ADD_NOISE
    fprintf('Input SNR (segment) ~ %.2f dB (mode=%s)\n', SNR_in_dB, NOISE_MODE);
else
    fprintf('Noise disabled.\n');
end

%% ---------------- FORM N x N HANKEL FROM NOISY SAMPLES ----------------
c = x_used_noisy(1:N);
r = x_used_noisy(N:(2*N-1));
H = hankel(c, r);                       % N x N (noisy Hankel)

% --------------- 2D UNITARY DFT ----------------
F = dftmtx_unitary(N);                  % F*F' = I
X = F * H * F';                         % 2D DFT of noisy Hankel
% ---------------- PCA (SVD) IN 2D-DFT DOMAIN ----------------
Xc = X - mean(X, 1);                    % column-center
[U,S,V] = svd(Xc, 'econ');
sing = diag(S);
var_expl = sing.^2 / sum(sing.^2);

if ~isempty(PCA_KEEP_K)
    K = min(PCA_KEEP_K, size(S,1));
elseif ~isempty(PCA_KEEP_VAR_FRAC)
    K = find(cumsum(var_expl) >= PCA_KEEP_VAR_FRAC, 1, 'first');
else
    K = size(S,1);
end
fprintf('Keeping K=%d PCs (variance captured = %.2f%%)\n', K, 100*sum(var_expl(1:K)));

Xc_hat = U(:,1:K) * S(1:K,1:K) * V(:,1:K)';  % low-rank (centered)
X_hat  = Xc_hat + mean(X, 1);                % add back means

%% ---------------- INVERSE 2D DFT + HANKELIZATION ----------------
H_hat = F' * X_hat * F;                 % denoised Hankel
x_rec = antiDiagAverage(H_hat);         % length Lh = 2N-1
x_rec = real(x_rec);

Lh = numel(x_rec);
xref_clean = x_used_clean(1:Lh);
xref_noisy = x_used_noisy(1:Lh);

% Metrics
mse_noisy = mean( (xref_noisy - xref_clean).^2 );
mse_rec   = mean( (x_rec      - xref_clean).^2 );

SNR_out_dB = 10*log10( var(xref_clean,1) / var(xref_clean - x_rec,1) );
fprintf('MSE noisy vs clean (first %d): %.6e\n', Lh, mse_noisy);
fprintf('MSE recon vs clean (first %d): %.6e\n', Lh, mse_rec);
fprintf('Output SNR ~ %.2f dB  (Improvement: %.2f dB)\n', SNR_out_dB, SNR_out_dB - SNR_in_dB);

%% ---------------- VISUALIZATION ----------------
% A) Continuous vs sampled (first segment)
t_seg_end = (Lh-1)/FS_SAMPLE;
idx_cont  = t_cont <= t_seg_end + (1/FS_CONT);
figure('Name','Continuous vs sampled (noisy)','NumberTitle','off','Position',[80 80 1150 320]);
plot(t_cont(idx_cont), x_cont(idx_cont), 'LineWidth', 1.2); hold on;
stem((0:Lh-1)/FS_SAMPLE, xref_noisy, 'filled');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
legend('high-rate (proxy continuous)','low-rate samples (noisy)');
title('Original high-rate signal and NOISY low-rate samples (aligned window)');

% B) Noisy samples vs PCA-Hankel reconstruction
figure('Name','Noisy vs Reconstruction','NumberTitle','off','Position',[80 420 1150 320]);
n_axis = 0:(M_used-1);
stairs(n_axis, x_used_noisy, 'LineWidth', 1.0); hold on;
stairs(0:(Lh-1), x_rec, '--', 'LineWidth', 1.5);
xlabel('Sample index n'); ylabel('Amplitude'); grid on;
legend(sprintf('noisy low-rate (first %d)', M_used), sprintf('reconstruction (len %d)', Lh));
title(sprintf('PCA-Hankel-DFT reconstruction — MSE=%.3e, SNR_in=%.2f dB, SNR_out=%.2f dB', ...
      mse_rec, SNR_in_dB, SNR_out_dB));

% C) Zoomed view (first zoom_samples points)
z = min([zoom_samples, Lh]);
figure('Name','Zoom (time-domain)','NumberTitle','off','Position',[80 760 1150 320]);
stairs(0:(z-1), xref_clean(1:z), 'LineWidth', 1.4); hold on;
stairs(0:(z-1), xref_noisy(1:z),  '-', 'LineWidth', 1.0);
stairs(0:(z-1), x_rec(1:z),     '--', 'LineWidth', 1.5);
xlabel('Sample index n'); ylabel('Amplitude'); grid on;
legend('clean (ref)','noisy','reconstruction');
title(sprintf('Zoomed segment (first %d samples)', z));

% D) Spectra (FFT) up to Nyquist (compare clean/noisy/recon)
Nfft  = 2^nextpow2(Lh);
Faxis = (0:Nfft-1)*(FS_SAMPLE/Nfft);
X_clean = fft(xref_clean, Nfft);
X_noisy = fft(xref_noisy, Nfft);
X_rec   = fft(x_rec,       Nfft);

figure('Name','Spectra','NumberTitle','off','Position',[1260 80 520 560]);
plot(Faxis(1:Nfft/2), abs(X_clean(1:Nfft/2)), 'LineWidth', 1.0); hold on;
plot(Faxis(1:Nfft/2), abs(X_noisy(1:Nfft/2)),  ':', 'LineWidth', 1.0);
plot(Faxis(1:Nfft/2), abs(X_rec(1:Nfft/2)),   '--', 'LineWidth', 1.4);
xlabel('Frequency (Hz)'); ylabel('|X(f)|'); grid on;
legend('clean segment','noisy','reconstruction');
title('Magnitude spectra (up to Nyquist)');

% E) PCA variance explained
figure('Name','PCA spectrum','NumberTitle','off','Position',[1260 660 520 320]);
stem(var_expl, 'filled'); xlim([1 size(S,1)]); grid on;
xlabel('PC index'); ylabel('Variance fraction'); title('PCA variance explained (2D-DFT domain)');

% F) Hankel magnitude: noisy vs denoised
figure('Name','Hankel magnitude','NumberTitle','off','Position',[1260 1020 520 320]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(abs(H));     axis image; colorbar; title('|H| (noisy)');
nexttile; imagesc(abs(H_hat)); axis image; colorbar; title('|H_{hat}| (denoised)');

disp('Done. Toggle ADD_NOISE, NOISE_MODE, SNR_dB/NOISE_SIGMA to experiment.');

%% ---------------- HELPER FUNCTIONS ----------------
function F = dftmtx_unitary(N)
% Unitary N-point DFT matrix (F*F' = I).
n = (0:N-1).'; k = 0:N-1;
F = exp(-1j*2*pi/N * (n*k)) / sqrt(N);
end

function x = antiDiagAverage(H)
% Diagonal averaging of an N x N Hankel-like matrix to a length-(2N-1) sequence.
[N, M] = size(H);
assert(N==M, 'H must be square');
L   = 2*N - 1;
x   = zeros(1, L);
cnt = zeros(1, L);
for i = 1:N
    for j = 1:N
        s = i + j - 1;       % 1..(2N-1)
        x(s)   = x(s) + H(i,j);
        cnt(s) = cnt(s) + 1;
    end
end
x = x ./ cnt;
x = x(:).';                  % row vector
end
%% ---------------- SINC-INTERPOLATION VISUALS (clean vs noisy vs PCA-recovered) ----------------
% Dense time grid matching the segment length used for reconstruction
Ts  = 1/FS_SAMPLE;
t_dense = (0:1/FS_CONT:(Lh-1)*Ts).';     % column vector

% Recompute ground-truth on t_dense directly (no slicing ambiguities)
x_true_dense = zeros(size(t_dense));
for k = 1:numel(component_freqs)
    x_true_dense = x_true_dense + component_amps(k) * ...
        sin(2*pi*component_freqs(k)*t_dense + component_phs(k));
end

% Ideal (bandlimited) reconstructions from the three sets of samples on the SAME grid
y_clean = sinc_interp(xref_clean(:), Ts, t_dense);   % from clean low-rate samples
y_noisy = sinc_interp(xref_noisy(:), Ts, t_dense);   % from noisy low-rate samples
y_pca   = sinc_interp(x_rec(:),       Ts, t_dense);  % from PCA-recovered samples

% ---- Overlay on dense grid, with ground-truth
figure('Name','Sinc-interpolated signals (dense grid)','NumberTitle','off','Position',[80 1100 1150 320]);
plot(t_dense, x_true_dense, 'LineWidth', 1.2); hold on;
plot(t_dense, y_clean, '-',  'LineWidth', 1.0);
plot(t_dense, y_noisy, ':',  'LineWidth', 1.0);
plot(t_dense, y_pca,   '--', 'LineWidth', 1.4);
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
legend('true @ dense grid','sinc from clean','sinc from noisy','sinc from PCA');
title('Ideal sinc interpolation from samples (clean / noisy / PCA)');

% ---- Zoomed view on dense grid
zoom_t = min( (zoom_samples-1)*Ts, t_dense(end) );
zi = (t_dense >= 0) & (t_dense <= zoom_t);

figure('Name','Zoom: Sinc-interpolated (dense grid)','NumberTitle','off','Position',[80 1440 1150 320]);
plot(t_dense(zi), x_true_dense(zi), 'LineWidth', 1.2); hold on;
plot(t_dense(zi), y_clean(zi), '-',  'LineWidth', 1.0);
plot(t_dense(zi), y_noisy(zi), ':',  'LineWidth', 1.0);
plot(t_dense(zi), y_pca(zi),   '--', 'LineWidth', 1.4);
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
legend('true','sinc from clean','sinc from noisy','sinc from PCA');
title(sprintf('Zoomed sinc interpolation (first ~%d samples worth)', zoom_samples));

% ---- Dense-grid errors vs true signal (same-length vectors guaranteed)
mse_clean_dense = mean( (y_clean - x_true_dense).^2 );
mse_noisy_dense = mean( (y_noisy - x_true_dense).^2 );
mse_pca_dense   = mean( (y_pca   - x_true_dense).^2 );

fprintf('Dense-grid MSE vs true:\n');
fprintf('  clean-samples sinc : %.6e\n', mse_clean_dense);
fprintf('  noisy-samples sinc : %.6e\n', mse_noisy_dense);
fprintf('  PCA-recov  sinc    : %.6e\n', mse_pca_dense);
%% ---------------- HELPER: ideal bandlimited interpolation ----------------
function y = sinc_interp(xn, Ts, t)
% Ideal bandlimited interpolation (sinc) from uniform samples xn at period Ts,
% evaluated on time grid t (column vector). Uses MATLAB's sinc: sinc(u) = sin(pi u)/(pi u).
% xn: column vector of length M, taken at n*Ts, n=0..M-1
% t : column vector of evaluation times
    M = numel(xn);
    n = (0:M-1).';                  % (M x 1)
    tau = (t.' - n*Ts) / Ts;        % (M x Nt), implicit expansion
    S = sinc(tau);                  % (M x Nt)
    y = (S.' * xn);                 % (Nt x 1)
end
