
clear; close all; clc;
%% ---------------- PARAMETERS ----------------
% Multi-tone signal
component_freqs = [300, 1200, 3000];   % Hz
component_amps  = [1.00, 0.60, 0.40];
component_phs   = [0.00, 0.3*pi, -0.2*pi];

FS_CONT  = 200e3;       % continuous (Hz)
DURATION = 0.02;        % seconds 

lambda = 1.2;          

% Actual sampling rate 
FS_SAMPLE = 6e3;       

% FFT options
Nfft   = 2^nextpow2(round(FS_SAMPLE*DURATION));  % zero-padding for smoother curves
use_hann_window = true;

zoom_t0 = 0.002; zoom_t1 = 0.006;

%% -------- CONSTRUCT SIGNALS ------
t_cont = 0:1/FS_CONT:DURATION-1/FS_CONT;

x_cont = zeros(size(t_cont));
for k = 1:numel(component_freqs)
    x_cont = x_cont + component_amps(k) * ...
        sin(2*pi*component_freqs(k)*t_cont + component_phs(k));
end

% Apply amplitude folding (nonlinear)
x_cont_fold = fold_centered(x_cont, lambda);

%% ---------------- SAMPLE BOTH SIGNALS ---------
t_samp = 0:1/FS_SAMPLE:DURATION-1/FS_SAMPLE;

% Sample original and folded (sample-and-hold ideal)
x_samp      = zeros(size(t_samp));
x_samp_fold = zeros(size(t_samp));
for k = 1:numel(component_freqs)
    x_samp      = x_samp      + component_amps(k) * sin(2*pi*component_freqs(k)*t_samp + component_phs(k));
    x_samp_fold = x_samp_fold + component_amps(k) * sin(2*pi*component_freqs(k)*t_samp + component_phs(k));
end

x_samp_fold = fold_centered(x_samp_fold, lambda);

%% --------- TIME-DOMAIN VISUALS -----
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
stem(t_samp(zi), x_samp_fold(zi), 'filled','MarkerFaceColor',[0.9 0.2 0.2]);
grid on; xlabel('Time (s)'); ylabel('Amplitude');
legend('original sampled','folded sampled');
title(sprintf('Sampled signals (zoom %.3f–%.3f s)', zoom_t0, zoom_t1));

%% --------- FFTs (properly normalized) ----
% One-sided magnitude spectra with windowing and coherent-gain normalization
[f, Xmag]    = onesided_spectrum(x_samp,      FS_SAMPLE, Nfft, use_hann_window);
[~, Xf_mag]  = onesided_spectrum(x_samp_fold, FS_SAMPLE, Nfft, use_hann_window);


figure('Name','FFT magnitude (sampled)','NumberTitle','off','Position',[60 740 1100 360]);
plot(f, Xmag, 'LineWidth', 1.3); hold on;
plot(f, Xf_mag, '--', 'LineWidth', 1.3);
grid on; xlim([0, FS_SAMPLE/2]);
xlabel('Frequency (Hz)'); ylabel('|X(f)| (approx amplitude)');
legend('original sampled','folded sampled','Location','best');
title('One-sided magnitude spectra (windowed & normalized)');

%% --------------- spectral “energy outside tones” metric ----------------
% This shows how folding injects energy beyond the original tones (harmonics/intermods)
tone_bins = round(component_freqs / (FS_SAMPLE/Nfft));   % expected bin indices (approx)
bin_tol   = 2;                                           % +/- bins considered "near tone"
mask_keep = false(size(f));
for b = tone_bins
    b = max(1, min(b, numel(f)));
    lo = max(1, b-bin_tol); hi = min(numel(f), b+bin_tol);
    mask_keep(lo:hi) = true;
end
% Energy outside tone neighborhoods
E_orig_out   = sum(Xmag(~mask_keep).^2);
E_folded_out = sum(Xf_mag(~mask_keep).^2);
fprintf('Energy outside tone neighborhoods:\n');
fprintf('  original sampled: %.3e\n', E_orig_out);
fprintf('  folded sampled  : %.3e (higher => more distortion from folding)\n', E_folded_out);

disp('Done. Try changing lambda, FS_SAMPLE, and component_freqs to see folding effects.');

%% ====================== Local helpers ======================

function y = fold_centered(x, lambda)
% Amplitude folding into [-lambda, +lambda)
% Equivalent to modulo on amplitude with symmetric wraparound.
    y = mod(x + lambda, 2*lambda) - lambda;
end

function [f, X1] = onesided_spectrum(x, fs, nfft, use_hann)
% Compute one-sided magnitude spectrum with optional Hann windowing,
% normalized for coherent gain so a pure tone's peak ~ its amplitude.
    x = x(:);
    N = numel(x);
    if use_hann
        w = hann(N, "periodic");
        cg = sum(w)/N;           % coherent gain of Hann
        xw = x .* w;
    else
        xw = x;
        cg = 1;
    end

    X = fft(xw, nfft) / (N*cg);  % normalize by coherent gain and length
    % One-sided
    X1 = abs(X(1:nfft/2+1));
    % For real signals, double non-DC/Nyquist bins to conserve energy
    if mod(nfft,2)==0
        X1(2:end-1) = 2*X1(2:end-1);
    else
        X1(2:end)   = 2*X1(2:end);
    end
    f = (0:(nfft/2))*(fs/nfft);
end
