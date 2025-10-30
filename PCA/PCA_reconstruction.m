
clear; close all; clc;

%% ----------- PARAMETERS -------
% Signal components
component_freqs = [300, 1200, 3000];   % Hz
component_amps  = [1.0, 0.6, 0.4];
component_phs   = [0, 0.3*pi, -0.2*pi];

FS_CONT = 200e3;    % "continuous" (Hz)
DUR     = 0.02;     % seconds

FS_SAMPLE  = 4e3;   % low/actual sampling rate (Hz) 
use_first_M = 120;  % number of low-rate samples used to build Hankel 

% Hankel size N
N_user = [];     

% PCA retention 
PCA_KEEP_K        = 6;    
PCA_KEEP_VAR_FRAC = [];     

zoom_samples = 80;       
%% ----- BUILD + SAMPLE -----
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
x_used  = x_low(1:M_used);

% Choose N with constraint 2N-1 <= M_used
if isempty(N_user)
    N = floor((M_used+1)/2);
else
    N = min(N_user, floor((M_used+1)/2));
end
assert(2*N-1 <= M_used, 'Need 2N-1 <= M_used.');

fprintf('Samples used: M=%d, Hankel size N=%d\n', M_used, N);

%% ----------- FORM N x N HANKEL -------
% Use 2N-1 consecutive samples for the square Hankel
c = x_used(1:N);
r = x_used(N:(2*N-1));
H = hankel(c, r);                      % N x N
% ---------------- 2D UNITARY DFT --------------
F = dftmtx_unitary(N);                 % F*F' = I
X = F * H * F';                        % 2D DFT of Hankel

% ---------------- PCA (SVD) IN 2D-DFT DOMAIN -----
% Column-center for PCA; complex SVD is fine
Xc = X - mean(X, 1);
[U, S, V] = svd(Xc, 'econ');
sing = diag(S);
var_expl = sing.^2 / sum(sing.^2);

if ~isempty(PCA_KEEP_K)
    K = min(PCA_KEEP_K, size(S,1));
elseif ~isempty(PCA_KEEP_VAR_FRAC)
    K = find(cumsum(var_expl) >= PCA_KEEP_VAR_FRAC, 1, 'first');
else
    K = size(S,1); % keep all
end
fprintf('Keeping K=%d PCs (variance captured = %.2f%%)\n', K, 100*sum(var_expl(1:K)));

Xc_hat = U(:,1:K) * S(1:K,1:K) * V(:,1:K)';  % low-rank approx (centered)
X_hat  = Xc_hat + mean(X, 1);                % add back column means

% ---------------- INVERSE 2D DFT + HANKELIZATION ----------------
H_hat = F' * X_hat * F;               % IDFT (unitary)
x_rec = antiDiagAverage(H_hat);       % length Lh = 2N-1 (complex possible)
x_rec = real(x_rec);                  % original is real; drop residual imag

Lh   = numel(x_rec);
xref = x_used(1:Lh);
mse  = mean( (x_rec(:) - xref(:)).^2 );
fprintf('Reconstructed length = %d, MSE vs reference (first %d): %.6e\n', Lh, Lh, mse);

%% ---------------- VISUALIZATION ----------------
% A) High-rate "continuous" vs low-rate samples (first segment shown)
t_seg_end = (Lh-1)/FS_SAMPLE;
idx_cont  = t_cont <= t_seg_end + (1/FS_CONT);
figure('Name','Continuous vs sampled (overview)','NumberTitle','off','Position',[80 80 1100 320]);
plot(t_cont(idx_cont), x_cont(idx_cont), 'LineWidth', 1.2); hold on;
stem((0:Lh-1)/FS_SAMPLE, xref, 'filled');
xlabel('Time (s)'); ylabel('Amplitude');
legend('Continuous)','Samples');
title('Original high-rate signal and low-rate samples (aligned window)'); grid on;

% B) Low-rate samples vs PCA-Hankel reconstruction (time overlay)
figure('Name','Samples vs reconstruction (time-domain)','NumberTitle','off','Position',[80 420 1100 320]);
n_axis = 0:(M_used-1);
stairs(n_axis, x_used, 'LineWidth', 1.2); hold on;
stairs(0:(Lh-1), x_rec, '--', 'LineWidth', 1.2);
xlabel('Sample index n'); ylabel('Amplitude');
legend(sprintf('original low-rate (first %d)', M_used), sprintf('reconstruction (len %d)', Lh));
title(sprintf('PCA-Hankel-DFT reconstruction — MSE (first %d) = %.3e, K=%d', Lh, mse, K)); grid on;

% D) Spectra (FFT) of low-rate original vs reconstruction
% Same sample rate (FS_SAMPLE) for both; compare magnitude up to Nyquist.
Nfft  = 2^nextpow2(Lh);
Xo    = fft(xref, Nfft);
Xh    = fft(x_rec, Nfft);
faxis = (0:Nfft-1)*(FS_SAMPLE/Nfft);
figure('Name','Spectra','NumberTitle','off','Position',[1220 80 520 560]);
plot(faxis(1:Nfft/2), abs(Xo(1:Nfft/2)), 'LineWidth', 1.1); hold on;
plot(faxis(1:Nfft/2), abs(Xh(1:Nfft/2)), '--', 'LineWidth', 1.1);
xlabel('Frequency (Hz)'); ylabel('|X(f)|'); grid on;
legend('original (segment)','reconstruction');
title('Magnitude spectra up to Nyquist');

% E) PCA variance explained
figure('Name','PCA spectrum','NumberTitle','off','Position',[1220 660 520 320]);
stem(var_expl, 'filled'); xlim([1 N]); grid on;
xlabel('PC index'); ylabel('Variance fraction'); title('PCA variance explained (2D-DFT domain)');

% F) Hankel & reconstructed Hankel magnitude
figure('Name','Hankel magnitude','NumberTitle','off','Position',[1220 1020 520 320]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(abs(H)); axis image; colorbar; title('|H|');
nexttile; imagesc(abs(H_hat)); axis image; colorbar; title('|H_{hat}|');

disp('Done. Tweak FS_SAMPLE, use_first_M, N_user, and PCA_KEEP_* to experiment.');

%% ---------------- HELPER FUNCTIONS ----------------
function F = dftmtx_unitary(N)
% Unitary N-point DFT matrix (F*F' = I).
n = (0:N-1).';
k = 0:N-1;
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